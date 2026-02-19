import cv2
import numpy as np
import mediapipe as mp

# ---------- SETTINGS ----------
DRAW_THICKNESS = 6
ERASE_SIZE = 40

COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
}

current_color_name = "blue"
current_color = COLORS[current_color_name]

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



canvas = None
prev_point = None

def finger_up(lm, tip, pip):
    # finger considered up if tip higher than pip joint
    return lm[tip].y < lm[pip].y

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mode = "NO HAND"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # fingertip position
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        # check gestures
        index_up = finger_up(lm, 8, 6)
        middle_up = finger_up(lm, 12, 10)

        # draw cursor
        cv2.circle(frame, (x, y), 10, current_color, -1)

        if index_up and not middle_up:
            # DRAW MODE
            mode = "DRAW"

            if prev_point is None:
                prev_point = (x, y)

            cv2.line(canvas, prev_point, (x, y), current_color, DRAW_THICKNESS)
            prev_point = (x, y)

        elif index_up and middle_up:
            # ERASE MODE
            mode = "ERASE"
            cv2.circle(canvas, (x, y), ERASE_SIZE, (0,0,0), -1)
            prev_point = None

        else:
            mode = "HOVER"
            prev_point = None

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        prev_point = None

    # combine drawing + video
    output = cv2.add(frame, canvas)

    # ---------- UI ----------
    cv2.rectangle(output, (0,0), (w,70), (0,0,0), -1)

    cv2.putText(output, f"Mode: {mode}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(output,
                f"Color: {current_color_name} | 1-4 change | C clear | Q quit",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    cv2.imshow("Finger Air Draw", output)

    key = cv2.waitKey(1) & 0xFF

    # ---------- KEYS ----------
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('1'):
        current_color_name = "blue"
        current_color = COLORS[current_color_name]
    elif key == ord('2'):
        current_color_name = "green"
        current_color = COLORS[current_color_name]
    elif key == ord('3'):
        current_color_name = "red"
        current_color = COLORS[current_color_name]
    elif key == ord('4'):
        current_color_name = "yellow"
        current_color = COLORS[current_color_name]

cap.release()
cv2.destroyAllWindows()
