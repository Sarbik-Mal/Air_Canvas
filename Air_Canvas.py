import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from tkinter import Tk, simpledialog

# Points for drawing
draw_points = [deque(maxlen=1024) for _ in range(12)]  # For 12 colors
color_indices = [0] * 12

# Define colors for the color bar
colors = [
    (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (128, 0, 128), (0, 128, 128),
    (128, 128, 0), (64, 64, 64), (192, 192, 192), (0, 0, 0)
]
colorIndex = 0

# Timer for color selection, clear action, and save action
color_message = ""
clear_message = ""
save_message = ""
message_start_time = None
color_selection_start_time = None
clear_start_time = None
save_start_time = None
color_message_display_time = None  # Timer for displaying the color picked message

# Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
cv2.resizeWindow('Paint', 800, 600)

# Function to get file name using Tkinter
def get_filename():
    root = Tk()
    root.withdraw()  # Hide the root window
    filename = simpledialog.askstring("Save Canvas", "Enter file name:")
    root.destroy()
    return filename

ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the color bar, clear button, and save button
    step = int(550 / len(colors))  # Adjust step size to fit the buttons
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i * step + 100, 0), ((i + 1) * step + 100, 33), color, -1)

    # Draw the clear button
    cv2.rectangle(frame, (0, 0), (50, 33), (200, 200, 200), -1)
    cv2.putText(frame, "CLR", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw the save button
    cv2.rectangle(frame, (50, 0), (100, 33), (100, 200, 100), -1)
    cv2.putText(frame, "SAVE", (60, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display any active messages
    if clear_message:
        cv2.putText(frame, clear_message, (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    elif save_message:
        cv2.putText(frame, save_message, (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    elif color_message and (time.time() - color_message_display_time <= 1):  # Show for 1 second
        cv2.putText(frame, color_message, (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, colors[colorIndex], 1, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post process the result
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger = (landmarks[8][0], landmarks[8][1])
            thumb = (landmarks[4][0], landmarks[4][1])
            center = fore_finger
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            # Check for pinch gesture
            distance = np.linalg.norm(np.array(thumb) - np.array(fore_finger))
            if distance < 30:  # If fingers are pinched, skip tracking
                draw_points[colorIndex].append(deque(maxlen=512))  # Create new deque
                color_indices[colorIndex] += 1
                continue

            # Check if the finger is on the clear button
            if 0 <= center[0] <= 50 and 0 <= center[1] <= 33:
                if clear_start_time is None:
                    clear_start_time = time.time()
                elif time.time() - clear_start_time >= 2:
                    paintWindow[:] = 255  # Clear the canvas
                    draw_points = [deque(maxlen=1024) for _ in range(12)]
                    color_indices = [0] * 12
                    clear_message = "Canvas Cleared!"
                    clear_start_time = None
                else:
                    clear_message = "Canvas will clear within 2 seconds"
            else:
                clear_start_time = None
                clear_message = ""

            # Check if the finger is on the save button
            if 50 <= center[0] <= 100 and 0 <= center[1] <= 33:
                if save_start_time is None:
                    save_start_time = time.time()
                elif time.time() - save_start_time >= 2:
                    filename = get_filename()
                    if filename:
                        cv2.imwrite(filename + ".png", paintWindow)
                        save_message = f"Canvas saved as {filename}.png!"
                    save_start_time = None
                else:
                    save_message = "Wait here for 2 seconds to save the canvas"
            else:
                save_start_time = None
                save_message = ""

            # Check if the finger is on the color bar
            if center[1] <= 33 and center[0] > 100:
                selected_color_index = (center[0] - 100) // step
                if color_selection_start_time is None:
                    color_selection_start_time = time.time()
                elif time.time() - color_selection_start_time >= 1:
                    colorIndex = selected_color_index
                    color_message = "The color has been picked"
                    color_message_display_time = time.time()  # Set the time when the message starts showing
                    color_selection_start_time = None
            else:
                color_selection_start_time = None
                color_message = ""

            # Add drawing logic
            if center[1] > 33:
                while len(draw_points[colorIndex]) <= color_indices[colorIndex]:
                    draw_points[colorIndex].append(deque(maxlen=512))
                draw_points[colorIndex][color_indices[colorIndex]].appendleft(center)

    else:
        draw_points[colorIndex].append(deque(maxlen=512))
        color_indices[colorIndex] += 1

    # Draw on the canvas
    for i, points in enumerate(draw_points):
        for j in range(len(points)):
            for k in range(1, len(points[j])):
                if points[j][k - 1] is None or points[j][k] is None:
                    continue
                cv2.line(frame, points[j][k - 1], points[j][k], colors[i], 2)
                cv2.line(paintWindow, points[j][k - 1], points[j][k], colors[i], 2)

    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Output', 800, 600)
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
