from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(data="dataset/data.yml", epochs=50, imgsz=640)

model.export(format="onnx") 