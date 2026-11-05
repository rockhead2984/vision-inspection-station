import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

sizes = [640, 480, 320, 160]   # YOLO 입력 해상도 후보
idx = 0                         # 현재 해상도 인덱스

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imgsz = sizes[idx]

    start = time.time()
    results = model(frame, imgsz=imgsz, verbose=False)   # 해상도 지정해서 검출
    elapsed = time.time() - start

    annotated = results[0].plot()

    # 검출된 물체 개수 + 종류
    names = [model.names[int(c)] for c in results[0].boxes.cls]
    count = len(names)

    fps = 1 / elapsed
    cv2.putText(annotated, f"imgsz: {imgsz}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated, f"Time: {elapsed*1000:.1f} ms  FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated, f"Objects: {count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("inspection", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):              # r 누르면 해상도 변경
        idx = (idx + 1) % len(sizes)
        print(f"해상도 변경 → {sizes[idx]}")

cap.release()
cv2.destroyAllWindows()