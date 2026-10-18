# roboflow-tiling

Roboflow-tiling: RF-DETR on aerial small objects, full-frame vs fixed tiles vs size-adaptive tiles.

VisDrone's drone images are full of objects a few pixels wide. RF-DETR, Roboflow's own open detector,
sees the whole frame at one resolution. Tiling feeds it crops so small objects get more pixels. The novel
piece is choosing the tile size per image from a first full-frame pass: images whose detections are tiny
get finer tiles, images with big objects skip tiling. No retraining anywhere.

## Results

<!-- results:start -->
_Run it to populate._
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
