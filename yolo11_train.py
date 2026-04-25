
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
# train parameters https://docs.ultralytics.com/ko/modes/train/#musgd-optimizer
