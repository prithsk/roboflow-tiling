"""roboflow-tiling: RF-DETR on aerial small objects, full-frame vs fixed tiles vs size-adaptive tiles.

VisDrone's drone images are full of objects a few pixels wide. RF-DETR, Roboflow's own open detector,
sees the whole frame at one resolution. Tiling feeds it crops so small objects get more pixels. The novel
piece is choosing the tile size per image from a first full-frame pass: images whose detections are tiny
get finer tiles, images with big objects skip tiling. No retraining anywhere.

  python bench.py --n 100          # downloads VisDrone2019-DET-val, runs on GPU if present, else CPU
"""
import argparse, io, zipfile, statistics, time, urllib.request, datetime as dt
from pathlib import Path
import numpy as np
from common import ROOT

VAL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
VIS2COCO = {1: "person", 2: "person", 3: "bicycle", 4: "car", 5: "car", 6: "truck", 9: "bus", 10: "motorcycle"}
CLASSES = sorted(set(VIS2COCO.values()))

def dataset():
    d = ROOT / "data"
    if not (d / "VisDrone2019-DET-val").exists():
        d.mkdir(exist_ok=True); zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(VAL, timeout=600).read())).extractall(d)
    root = d / "VisDrone2019-DET-val"; out = []
    for img in sorted((root / "images").glob("*.jpg")):
        gt = []
        for l in (root / "annotations" / (img.stem + ".txt")).read_text().splitlines():
            x, y, w, h, s, c = map(int, l.split(",")[:6])
            if c in VIS2COCO and w > 0 and h > 0: gt.append((x, y, x + w, y + h, VIS2COCO[c]))
        out.append((img, gt))
    return out

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i; return i / u if u > 0 else 0.0

def nms(dets, thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(d[5] != k[5] or iou(d, k) < thr for k in keep): keep.append(d)
    return keep

def tiles(W, H, size, overlap=0.2):
    step = max(1, int(size * (1 - overlap))); xs = list(range(0, max(1, W - size), step)) + [max(0, W - size)]
    ys = list(range(0, max(1, H - size), step)) + [max(0, H - size)]
    return sorted({(x, y, min(W, x + size), min(H, y + size)) for x in xs for y in ys})

def ap50(preds, gts, cls, min_area=0, max_area=1e12):
    """VOC-style AP at IoU 0.5 for one class, optionally restricted to GT boxes in an area band."""
    gt = {i: [g for g in G if g[4] == cls and min_area <= (g[2] - g[0]) * (g[3] - g[1]) < max_area] for i, G in enumerate(gts)}
    npos = sum(len(v) for v in gt.values())
    if not npos: return None
    allp = sorted(((i, p) for i, P in enumerate(preds) for p in P if p[5] == cls), key=lambda t: -t[1][4])
    used = {i: [False] * len(v) for i, v in gt.items()}; tp = []
    for i, p in allp:
        best, j = 0.0, -1
        for k, g in enumerate(gt[i]):
            o = iou(p, g)
            if o > best: best, j = o, k
        if best >= 0.5 and not used[i][j]: used[i][j] = True; tp.append(1)
        else: tp.append(0)
    if not tp: return 0.0
    c = np.cumsum(tp); rec = c / npos; prec = c / np.arange(1, len(tp) + 1)
    return float(sum(max([p for p, r in zip(prec, rec) if r >= t] or [0]) for t in np.linspace(0, 1, 11)) / 11)

def detector():
    from rfdetr import RFDETRBase
    from rfdetr.util.coco_classes import COCO_CLASSES
    return RFDETRBase(), COCO_CLASSES

def predict(model, names, img, box=None, thr=0.2):
    crop = img.crop(box) if box else img; ox, oy = (box[0], box[1]) if box else (0, 0)
    d = model.predict(crop, threshold=thr); out = []
    for (x1, y1, x2, y2), s, c in zip(d.xyxy, d.confidence, d.class_id):
        n = names.get(int(c)) if isinstance(names, dict) else names[int(c)]
        if n in CLASSES: out.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(s), n))
    return out

def adaptive_size(full_dets, W, H):
    """Pick a tile size from how big the first-pass detections are. Big objects: no tiling."""
    if not full_dets: return 640
    med = statistics.median(((d[2] - d[0]) * (d[3] - d[1])) ** 0.5 for d in full_dets)
    if med >= 96: return None
    if med >= 48: return 960
    return 640 if med >= 24 else 480

def main(n):
    from PIL import Image
    model, names = detector(); data = dataset()[:n]; preds = {"full": [], "fixed_640": [], "adaptive": []}; gts = []; times = {k: 0.0 for k in preds}
    for k, (path, gt) in enumerate(data, 1):
        img = Image.open(path).convert("RGB"); W, H = img.size; gts.append(gt)
        t = time.perf_counter(); full = predict(model, names, img); times["full"] += time.perf_counter() - t; preds["full"].append(full)
        t = time.perf_counter(); fx = full + [d for b in tiles(W, H, 640) for d in predict(model, names, img, b)]
        preds["fixed_640"].append(nms(fx)); times["fixed_640"] += time.perf_counter() - t + times["full"] / k
        t = time.perf_counter(); size = adaptive_size(full, W, H)
        ad = full + ([d for b in tiles(W, H, size) for d in predict(model, names, img, b)] if size else [])
        preds["adaptive"].append(nms(ad)); times["adaptive"] += time.perf_counter() - t + times["full"] / k
        if k % 10 == 0: print(f"  {k}/{len(data)}", flush=True)
    def mAP(P, lo=0, hi=1e12):
        v = [x for x in (ap50(P, gts, c, lo, hi) for c in CLASSES) if x is not None]; return sum(v) / len(v)
    rows = [(k, mAP(P), mAP(P, 0, 32 ** 2), times[k] / len(data)) for k, P in preds.items()]
    md = [f"# RF-DETR tiling on VisDrone2019-DET-val, {dt.date.today()}", "",
          f"{len(data)} images. RF-DETR Base, COCO weights, no retraining. VisDrone classes mapped to COCO: {', '.join(CLASSES)}.", "",
          "| Inference | mAP50 | mAP50, small objects (<32x32) | Seconds per image |", "|---|---|---|---|"]
    md += [f"| {k} | {a:.3f} | {s:.3f} | {t:.2f} |" for k, a, s, t in rows]
    md += ["", "## Honest limits", "",
           "- COCO weights on drone imagery. A fine-tuned RF-DETR would score far higher; this measures what tiling adds to the same weights.",
           "- 11-point VOC AP at IoU 0.5, implemented here, not COCO's 101-point AP. Use it to compare rows, not to compare with papers.",
           "- The adaptive rule has four size buckets set by hand. It shows the idea pays off or doesn't, not a tuned policy."]
    (ROOT / "results").mkdir(exist_ok=True); s = "\n".join(md); (ROOT / "results" / f"{dt.date.today()}.md").write_text(s, encoding="utf-8"); print(s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); main(ap.parse_args().n)
