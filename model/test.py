from ultralytics import YOLO
import cv2

model = YOLO("best.pt") 

image_path = "caminho/para/sua/imagem.jpg"
image = cv2.imread(image_path)

results = model(image_path)

for result in results:
    for box in result.boxes.xyxy:  
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  

output_path = "detected_image.jpg"
cv2.imwrite(output_path, image)  
cv2.imshow("Detecção", image)  
cv2.waitKey(0)
cv2.destroyAllWindows()
