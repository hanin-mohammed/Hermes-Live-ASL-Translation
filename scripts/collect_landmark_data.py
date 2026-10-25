import cv2
import mediapipe as mp
import csv
from pathlib import Path

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

project_folder = Path(__file__).resolve().parent.parent
hand_model_file = str(project_folder / "models" / "hand_landmarker.task")
data_file = project_folder / "data" / "landmark_data.csv"
sign_labels = ["A", "B", "C", "1", "2", "3", "ILY"]
sign_counts = {
    "A": 0,
    "B": 0,
    "C": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "ILY": 0
}

try:
    with open(data_file, "r") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) == 64:
                label = row[63]

                if label in sign_counts:
                    sign_counts[label] = sign_counts[label] + 1
except FileNotFoundError:
    print("No old data file found. Starting counts from zero.")


hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_file),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()


while True:
    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break


    else:
        flipped_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        mediapipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = hand_detector.detect(mediapipe_image)

        landmark_row = []

        if not result.hand_landmarks:
            cv2.putText(flipped_frame, "No hand detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            first_hand = result.hand_landmarks[0]

            for landmark in first_hand:
                landmark_row.append(landmark.x)
                landmark_row.append(landmark.y)
                landmark_row.append(landmark.z)



            cv2.putText(flipped_frame, "Press A B C 1 2 3 or I", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        count_line_1 = "A:" + str(sign_counts["A"]) + "  B:" + str(sign_counts["B"]) + "  C:" + str(sign_counts["C"])
        count_line_2 = "1:" + str(sign_counts["1"]) + "  2:" + str(sign_counts["2"]) + "  3:" + str(sign_counts["3"]) + "  ILY:" + str(sign_counts["ILY"])

        cv2.putText(flipped_frame, count_line_1, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(flipped_frame, count_line_2, (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Collect Landmark Data", flipped_frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        sign_label = ""

        if key == ord("a"):
            sign_label = "A"

        if key == ord("b"):
            sign_label = "B"

        if key == ord("c"):
            sign_label = "C"

        if key == ord("1"):
            sign_label = "1"

        if key == ord("2"):
            sign_label = "2"

        if key == ord("3"):
            sign_label = "3"

        if key == ord("i"):
            sign_label = "ILY"

        if sign_label != "":
            if len(landmark_row) == 63:
                landmark_row.append(sign_label)

                with open(data_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(landmark_row)

                sign_counts[sign_label] = sign_counts[sign_label] + 1

                print("Saved", sign_label)
                print("Current count:", sign_counts[sign_label])
            else:
                print("No hand detected. Nothing saved.")

camera.release()
cv2.destroyAllWindows()
