from ultralytics import YOLO

model = YOLO("model/best.pt")

metrics = model.val()
print(metrics)  
