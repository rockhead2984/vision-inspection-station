import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 흑백 변환: 컬러 3채널 → 1채널 (연산량 1/3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 블러: 노이즈 제거 (5x5 영역 평균)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 엣지 검출: 밝기가 급격히 바뀌는 경계선 찾기
    edges = cv2.Canny(blur, 100, 200)

    # 4. 윤곽선 검출: 엣지로 둘러싸인 물체 외곽 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 일정 크기 이상만 박스로 표시 (작은 잡음 무시)
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

    cv2.putText(frame, f"Objects: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("result", frame)
    cv2.imshow("edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
