1. 검증 코드
from ultralytics import YOLO

model = YOLO("yolov11n.pt")

results = model.val(
    data="coco128.yaml",
    imgsz=640,
    batch=8,
    device=0
)

print(results.box.map)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)


#runs/detect/val/

#주요 파일:

#confusion_matrix.png
#PR_curve.png
#F1_curve.png
#P_curve.png
#R_curve.png


