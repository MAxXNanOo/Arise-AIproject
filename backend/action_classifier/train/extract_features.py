import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO

# คอนฟิกเส้นทางโฟลเดอร์
MODEL_PATH = "yolo11n-pose.pt"
RAW_DIR = "raw_gru_dataset"  # 📁 เปลี่ยนเป็นโฟลเดอร์หลักของชุดข้อมูล RAW
OUTPUT_DIR = "gru_dataset"

model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FRAMES = 30  # จำนวนเฟรมคงที่สำหรับ GRU

# ค้นหาไฟล์วิดีโอทั้งหมดในโฟลเดอร์ย่อย (.mp4 และ .avi)
video_files = glob.glob(os.path.join(RAW_DIR, "*", "*.mp4")) + glob.glob(os.path.join(RAW_DIR, "*", "*.avi"))
print(f"🔄 พบไฟล์วิดีโอทั้งหมด {len(video_files)} ไฟล์ ในชุดข้อมูล RAW")

for video_path in video_files:
    # ดึงชื่อคลาส (ชื่อโฟลเดอร์ก่อนหน้าไฟล์) และชื่อไฟล์
    path_parts = video_path.split(os.sep)
    action_class = path_parts[-2]  # เช่น running, walking, clapping
    video_name = os.path.splitext(path_parts[-1])[0]
    
    print(f"🎬 ประมวลผลคลาส [{action_class}] -> คลิป: {video_name}...")
    
    cap = cv2.VideoCapture(video_path)
    track_sequences = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ใช้ .track เพื่อติดตามบุคคลในมุมกล้องวงจรปิด
        results = model.track(frame, persist=True, verbose=False)
        if len(results) == 0:
            continue

        result = results[0]
        if result.keypoints is None or result.boxes.id is None:
            continue

        keypoints_list = result.keypoints.xy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        h, w = frame.shape[:2]

        for i, track_id in enumerate(track_ids):
            person_kpts = keypoints_list[i].copy()
            if track_id not in track_sequences:
                track_sequences[track_id] = []

            # Normalize พิกัด
            person_kpts[:, 0] /= w
            person_kpts[:, 1] /= h
            track_sequences[track_id].append(person_kpts)

    cap.release()

    # บันทึกข้อมูลและปรับแต่งขนาดเฟรมให้คงที่
    if len(track_sequences) > 0:
        for track_id, sequence in track_sequences.items():
            sequence_np = np.array(sequence, dtype=np.float32)
            current_frames = sequence_np.shape[0]
            
            # การจัดการมิติให้คงที่ (Padding / Truncating)
            if current_frames >= TARGET_FRAMES:
                sequence_np = sequence_np[:TARGET_FRAMES]
            else:
                padding_size = TARGET_FRAMES - current_frames
                padding = np.zeros((padding_size, 17, 2), dtype=np.float32)
                sequence_np = np.vstack((sequence_np, padding))
            
            # Flatten พิกัดจาก (30, 17, 2) เป็น (30, 34)
            sequence_np = sequence_np.reshape(TARGET_FRAMES, -1)
            
            # ตั้งชื่อไฟล์ให้มีชื่อคลาสรวมอยู่ด้วยเพื่อสกัด Label ตอนเทรน
            file_name = f"{action_class}_{video_name}_person_{track_id}.npy"
            np.save(os.path.join(OUTPUT_DIR, file_name), sequence_np)
    else:
        print(f"   ⚠️ ไม่พบบุคคลในไฟล์ {video_name}")

print("\n🎉 สกัดฟีเจอร์พิกัดท่าทางจาก RAW เสร็จสิ้น!")
