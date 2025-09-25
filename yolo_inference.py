from ultralytics import YOLO

model  = YOLO("yolov8x")

results = model.predict('input_video/08fd33_4.mp4', save=True)
print(results[0])

for box in results[0].boxes:
    print(box.xyxy)  # x1, y1, x2, y2
    print(box.cls)   # class
    print(box.conf)  # confidence

# results[0].show()  # display the result