import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    ok = cap.isOpened()
    print(i, "OPEN" if ok else "NO")
    if ok:
        ret, frame = cap.read()
        print("   read:", ret, "shape:", None if frame is None else frame.shape)
    cap.release()
