# roboflow-tiling

Roboflow-tiling: RF-DETR on aerial small objects, full-frame vs fixed tiles vs size-adaptive tiles.

VisDrone's drone images are full of objects a few pixels wide. RF-DETR, Roboflow's own open detector,
sees the whole frame at one resolution. Tiling feeds it crops so small objects get more pixels. The novel
piece is choosing the tile size per image from a first full-frame pass: images whose detections are tiny
get finer tiles, images with big objects skip tiling. No retraining anywhere.

## Results

<!-- results:start -->
# RF-DETR tiling on VisDrone2019-DET-val, 2026-10-18

100 images. RF-DETR Base, COCO weights, no retraining. VisDrone classes mapped to COCO: bicycle, bus, car, motorcycle, person, truck.

| Inference | mAP50 | mAP50, small objects (<32x32) | Seconds per image |
|---|---|---|---|
| full | 0.345 | 0.092 | 0.86 |
| fixed_640 | 0.403 | 0.122 | 4.60 |
| adaptive | 0.427 | 0.162 | 4.99 |

## Honest limits

- COCO weights on drone imagery. A fine-tuned RF-DETR would score far higher; this measures what tiling adds to the same weights.
- 11-point VOC AP at IoU 0.5, implemented here, not COCO's 101-point AP. Use it to compare rows, not to compare with papers.
- The adaptive rule has four size buckets set by hand. It shows the idea pays off or doesn't, not a tuned policy.
<!-- results:end -->

## Run it

```
pip install -r requirements.txt
cp .env.example .env    # keys: none
python bench.py --n 100
```

Every run writes `results/<date>.md` with the numbers and an honest-limits section.

## License

MIT
