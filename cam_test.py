import cv2

cap = cv2.VideoCapture(0)   # 0번 카메라 = 노트북 웹캠

while True:
    ret, frame = cap.read()  # 한 프레임 읽기
    if not ret:
        break
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()