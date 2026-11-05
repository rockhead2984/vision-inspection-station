import cv2
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

IMGSZ = 480          # 성능 측정으로 찾은 최적값
CONF_MIN = 0.5       # 신뢰도 50% 미만 검출은 무시

# 검사 기준: 물체별 허용 길이 (픽셀, 긴 변 기준)
# 캘리브레이션(2026-11-05): 핸드폰 272px, 마우스 154~156px 측정 → 기준값 ±10%
# ※ 카메라 높이가 바뀌면 픽셀 길이도 바뀌므로 다시 측정해야 함
SPEC = {
    "cell phone": (245, 300),
    "mouse":      (140, 170),
}

LOG_FILE = "inspection_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["time", "object", "conf", "length_px", "result", "reason"])

total, fail = 0, 0


def measure_length(crop):
    """OpenCV로 물체 긴 변 길이 측정 (흰 배경 기준)"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu: 임계값 자동 결정, INV: 어두운 물체를 흰색으로
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    c = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)   # 회전된 물체도 실제 길이로 측정
    return max(w, h)


def judge(name, length):
    """판정 로직"""
    if name not in SPEC:
        return "FAIL", "unregistered"
    lo, hi = SPEC[name]
    if lo <= length <= hi:
        return "PASS", "ok"
    return "FAIL", "size"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    # ROI: 화면 가운데 절반만 검사 영역
    x1, y1, x2, y2 = W // 4, H // 4, W * 3 // 4, H * 3 // 4
    roi = frame[y1:y2, x1:x2]

    start = time.time()
    results = model(roi, imgsz=IMGSZ, verbose=False)
    elapsed = (time.time() - start) * 1000

    best = None   # 이번 프레임에서 가장 확실한 검출
    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < CONF_MIN:
            continue
        name = model.names[int(box.cls)]
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        crop = roi[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue

        length = measure_length(crop)
        result, reason = judge(name, length)
        color = (0, 200, 0) if result == "PASS" else (0, 0, 255)

        # ROI 좌표 → 원본 프레임 좌표로 변환해서 그리기
        cv2.rectangle(frame, (x1 + bx1, y1 + by1), (x1 + bx2, y1 + by2), color, 2)
        cv2.putText(frame, f"{name} {length:.0f}px {result}",
                    (x1 + bx1, y1 + by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if best is None or conf > best[1]:
            best = (name, conf, length, result, reason)

    # 검사 영역 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    rate = fail / total * 100 if total else 0
    cv2.putText(frame, f"Time: {elapsed:.1f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Total: {total}  Fail: {fail}  Rate: {rate:.1f}%", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE: inspect  Q: quit", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("station", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and best:     # 스페이스바 = 검사 확정 + 기록
        name, conf, length, result, reason = best
        total += 1
        if result == "FAIL":
            fail += 1
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    name, f"{conf:.2f}", f"{length:.0f}", result, reason])
        print(f"[{total}] {name} {length:.0f}px → {result} ({reason})")

cap.release()
cv2.destroyAllWindows()