import cv2
import mediapipe as mp
import numpy as np

# FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh

# Outer lip boundary
OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
              291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# Inner lip boundary
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

LIP_COLORS = {
    '1': (40, 50, 200),   # Soft red
    '2': (90, 60, 180),   # Natural pink
    '3': (80, 80, 145),   # Mauve rose
    '4': (110, 50, 150),  # Plum tint
    '5': (70, 40, 100),   # Nude brown
    '6': (30, 30, 30),    # Dark subtle
    '7': (160, 150, 150), # Soft nude
    '8': (140, 90, 200),  # Soft fuchsia
    '9': (160, 120, 200), # Peach rose
    '10' : (0, 0, 220),   # Deeper red
    '11': (120, 40, 200), # Deeper pink
    '12': (0, 90, 200),   # Coral red
    '13': (140, 0, 160),  # Plum
    '14': (60, 30, 120),  # Nude brown
    '15': (15, 15, 15),   # Deep black
    '16': (255, 20, 200), # Barbie pink (tronger nudebrighter)
    '17': (210, 150, 200) # Peach rose

}

current_color = '14'

def main():
    global current_color

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Create face mesh model
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            h, w, _ = frame.shape

            # MediaPipe expects RGB images
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # If face detected
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # -------------------- LIPSTICK OVERLAY --------------------

                # 1. Collect outer lip points
                outer_points = []
                for idx in OUTER_LIPS:
                     lm = face_landmarks.landmark[idx]
                     x, y = int(lm.x * w), int(lm.y * h)
                     outer_points.append([x, y])
                outer_points = np.array(outer_points, np.int32)

                # 2. Collect inner lip points (mouth opening)
                inner_points = []
                for idx in INNER_LIPS:
                     lm = face_landmarks.landmark[idx]
                     x, y = int(lm.x * w), int(lm.y * h)
                     inner_points.append([x, y])
                inner_points = np.array(inner_points, np.int32)

                # 3. Create a mask
                lip_mask = np.zeros_like(frame)

                # Fill outer lips with red color
                cv2.fillPoly(lip_mask, [outer_points], LIP_COLORS[current_color])

                # Remove inner lips (this prevents teeth coloring)
                cv2.fillPoly(lip_mask, [inner_points], (0, 0, 0))

                lip_mask = cv2.GaussianBlur(lip_mask, (9, 9), 3)

                # Blend with original frame
                alpha = 0.25
                frame = cv2.addWeighted(frame, 1 - alpha, lip_mask, alpha, 0)

                # gloss effect: add slight highlight on upper lip
                highlight = lip_mask.copy()
                highlight = cv2.GaussianBlur(highlight, (51, 51), 30)
                frame = cv2.addWeighted(frame, 1, highlight, 0.1, 0)

                # Natural gradient: darken inner lip slightly
                inner_mask = np.zeros_like(frame)
                cv2.fillPoly(inner_mask, [inner_points], (20, 20, 20))
                inner_mask = cv2.GaussianBlur(inner_mask, (9, 9), 5)
                frame = cv2.addWeighted(frame, 1, inner_mask, 0.15, 0)

                # ------------------ END LIPSTICK OVERLAY ------------------   

            # Show output
            cv2.imshow("AI Stylist - Virtual Lipstick", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # Switch lip colors
            if chr(key) in LIP_COLORS:
                current_color = chr(key)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

