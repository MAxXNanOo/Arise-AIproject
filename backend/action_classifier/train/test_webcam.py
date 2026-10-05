import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from collections import deque
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

YOLO_MODEL = "yolo11n-pose.pt"
GRU_MODEL = "training_results/best_action_gru.pth"
DATA_DIR = "gru_dataset"

SEQUENCE_LENGTH = 30
NUM_CLASSES = 7


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using: {device}")


# ============================================================
# GET CLASS NAMES FROM DATASET
# ============================================================

file_paths = glob.glob(
    os.path.join(DATA_DIR, "*.npy")
)

if len(file_paths) == 0:
    print("ERROR: ไม่พบไฟล์ .npy ใน gru_dataset")
    exit()


# ใช้วิธีเดียวกับ dataset_model.py
class_names = sorted(
    list(
        set(
            os.path.basename(f).split("_")[0]
            for f in file_paths
        )
    )
)

print("\nClasses detected:")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print(f"\nTotal classes: {len(class_names)}")


if len(class_names) != NUM_CLASSES:

    print(
        f"\nERROR: Dataset มี {len(class_names)} classes "
        f"แต่โมเดลถูกสร้างมาสำหรับ {NUM_CLASSES} classes"
    )

    exit()


# ============================================================
# GRU MODEL
# ============================================================

class ActionGRU(nn.Module):

    def __init__(
        self,
        input_size=34,
        hidden_size=64,
        num_layers=2,
        num_classes=7
    ):
        super(ActionGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # x:
        # (Batch, 30, 34)

        out, _ = self.gru(x)

        # ใช้ frame สุดท้าย
        out = out[:, -1, :]

        # Classification
        out = self.fc(out)

        return out


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO Pose model...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# LOAD GRU
# ============================================================

print("\nLoading GRU model...")

model = ActionGRU(
    input_size=34,
    hidden_size=64,
    num_layers=2,
    num_classes=NUM_CLASSES
).to(device)


# โหลด weights
state_dict = torch.load(
    GRU_MODEL,
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()

print("GRU loaded successfully.")


# ============================================================
# CAMERA
# ============================================================

print("\nOpening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Cannot open camera.")

    exit()


print("Camera started.")
print("Press Q to quit.")


# ============================================================
# SEQUENCE BUFFER
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


# Prediction ล่าสุด
current_action = "Waiting..."

current_confidence = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Cannot read camera.")

        break


    # ========================================================
    # YOLO POSE
    # ========================================================

    results = yolo(
        frame,
        verbose=False
    )


    # ========================================================
    # CHECK PERSON / KEYPOINT
    # ========================================================

    if (
        len(results) > 0
        and results[0].keypoints is not None
        and len(results[0].keypoints) > 0
    ):

        # ----------------------------------------------------
        # เอาคนแรกที่ตรวจพบ
        # ----------------------------------------------------

        keypoints = results[0].keypoints.xy[0]


        # ----------------------------------------------------
        # YOLO Pose มี 17 keypoints
        # ----------------------------------------------------

        if keypoints.shape[0] >= 17:

            # เอาเฉพาะ X,Y
            keypoints = keypoints[:17, :2]

            # แปลงเป็น numpy
            keypoints = keypoints.cpu().numpy()


            # ------------------------------------------------
            # 17 keypoints × 2
            # = 34 features
            # ------------------------------------------------

            features = keypoints.flatten()


            # ------------------------------------------------
            # เพิ่มเข้า sequence
            # ------------------------------------------------

            if len(features) == 34:

                sequence.append(features)


    # ========================================================
    # GRU PREDICTION
    # ========================================================

    if len(sequence) == SEQUENCE_LENGTH:

        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        sequence_array = np.array(
            sequence,
            dtype=np.float32
        )


        # shape:
        # (30, 34)


        # ----------------------------------------------------
        # เพิ่ม Batch dimension
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            sequence_array,
            dtype=torch.float32
        ).unsqueeze(0).to(device)


        # shape:
        # (1, 30, 34)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        class_index = prediction.item()

        current_confidence = (
            confidence.item() * 100
        )


        # ----------------------------------------------------
        # Get class name
        # ----------------------------------------------------

        if class_index < len(class_names):

            current_action = class_names[
                class_index
            ]

        else:

            current_action = "Unknown"


    # ========================================================
    # DRAW YOLO RESULT
    # ========================================================

    annotated_frame = results[0].plot()


    # ========================================================
    # DISPLAY ACTION
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Action: {current_action}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY CONFIDENCE
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Confidence: {current_confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY SEQUENCE
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "YOLO Pose + GRU Action Recognition",
        annotated_frame
    )


    # ========================================================
    # PRESS Q TO QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEAN UP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nProgram finished.")