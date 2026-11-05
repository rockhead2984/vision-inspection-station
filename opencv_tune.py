import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("control")
cv2.createTrackbar("Canny Low", "control", 50, 255, nothing)
cv2.createTrackbar("Canny High", "control", 150, 255, nothing)
cv2.createTrackbar("Min Area", "control", 1000, 20000, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    low = cv2.getTrackbarPos("Canny Low", "control")
    high = cv2.getTrackbarPos("Canny High", "control")
    min_area = cv2.getTrackbarPos("Min Area", "control")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
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