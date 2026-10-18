import bench
assert bench.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and bench.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
k = bench.nms([(0, 0, 10, 10, 0.9, "car"), (1, 1, 10, 10, 0.8, "car"), (1, 1, 10, 10, 0.8, "person")]); assert len(k) == 2
t = bench.tiles(1360, 765, 640); assert all(b[2] - b[0] <= 640 for b in t) and max(b[2] for b in t) == 1360 and max(b[3] for b in t) == 765
gts = [[(0, 0, 10, 10, "car"), (50, 50, 60, 60, "car")]]
assert bench.ap50([[(0, 0, 10, 10, 0.9, "car"), (50, 50, 60, 60, 0.8, "car")]], gts, "car") == 1.0
assert bench.ap50([[(0, 0, 10, 10, 0.9, "car")]], gts, "car") < 0.6
assert bench.adaptive_size([(0, 0, 200, 200, 0.9, "car")], 1000, 1000) is None and bench.adaptive_size([(0, 0, 10, 10, 0.9, "car")], 1000, 1000) == 480
print("all tests passed")
