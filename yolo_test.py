import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # n = nano, 제일 가벼운 모델
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)   # 검출
    annotated = results[0].plot()            # 박스+라벨 그린 이미지

    cv2.imshow("yolo", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()