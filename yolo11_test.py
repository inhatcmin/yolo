from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
results = model(["test1.jpg","test2.jpg"])  # return a list of Results objects

#results =model(
#    source="test_images/",   # 이미지 폴더
#    conf=0.25,
 #   save=True,
  #  device=0
#)


# Process results list
for i, result in enumerate(results):

    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    filename = f"result_{i}.jpg"
    result.save(filename=filename)
    print(f"Saved: {filename}")