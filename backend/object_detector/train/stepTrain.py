from ultralytics import YOLO
from pathlib import Path


def main():
    # =========================================================
    # ตั้งค่า
    # =========================================================

    # Dataset
    DATA_YAML = r"dataset/data.yaml"

    # โมเดลเริ่มต้นสำหรับรอบที่ 1
    INITIAL_MODEL = "yolo11n.pt"

    # จำนวนรอบ
    TOTAL_ROUNDS = 4

    # จำนวน epoch ต่อรอบ
    EPOCHS_PER_ROUND = 50

    # ขนาดภาพ
    IMGSZ = 640

    # Batch
    # -1 = ให้ Ultralytics หา Batch ที่เหมาะกับ GPU อัตโนมัติ
    BATCH = -1

    # จำนวน worker
    WORKERS = 4

    # GPU
    DEVICE = 0

    # โฟลเดอร์หลักเก็บผลลัพธ์
    PROJECT = "training_results"

    # =========================================================
    # โมเดลที่จะใช้ในแต่ละรอบ
    # =========================================================

    current_model = INITIAL_MODEL

    # =========================================================
    # เริ่ม Train 6 รอบ
    # =========================================================

    for round_number in range(1, TOTAL_ROUNDS + 1):

        print("\n" + "=" * 80)
        print(f"START ROUND {round_number}/{TOTAL_ROUNDS}")
        print("=" * 80)

        round_name = f"round-{round_number}"

        print(f"Model  : {current_model}")
        print(f"Epochs : {EPOCHS_PER_ROUND}")
        print(f"Output : {PROJECT}/{round_name}")

        # -----------------------------------------------------
        # โหลดโมเดล
        # -----------------------------------------------------

        model = YOLO(current_model)

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        model.train(
            data=DATA_YAML,

            # 50 epochs ต่อรอบ
            epochs=EPOCHS_PER_ROUND,

            # image size
            imgsz=IMGSZ,

            # batch
            batch=BATCH,

            # workers
            workers=WORKERS,

            # GPU
            device=DEVICE,

            # โฟลเดอร์ผลลัพธ์
            project=PROJECT,
            name=round_name,

            # บันทึกโมเดล
            save=True,

            # สร้างกราฟ
            plots=True,

            # ใช้ validation
            val=True,

            # ให้ train ครบ 50 epochs
            patience=EPOCHS_PER_ROUND,

            verbose=True
        )

        # -----------------------------------------------------
        # อ่าน path จริงที่ Ultralytics บันทึก
        # -----------------------------------------------------

        save_dir = Path(model.trainer.save_dir)

        best_model = save_dir / "weights" / "best.pt"
        last_model = save_dir / "weights" / "last.pt"

        print("\n" + "-" * 80)
        print(f"ROUND {round_number} COMPLETED")
        print(f"Save directory : {save_dir}")
        print(f"Best model     : {best_model}")
        print(f"Last model     : {last_model}")
        print("-" * 80)

        # -----------------------------------------------------
        # ตรวจสอบ best.pt
        # -----------------------------------------------------

        if not best_model.exists():
            print("\nERROR: ไม่พบ best.pt")
            print("หยุดการ Train")

            weights_dir = save_dir / "weights"

            if weights_dir.exists():
                print("\nไฟล์ในโฟลเดอร์ weights:")
                for file in weights_dir.iterdir():
                    print(f" - {file}")

            break

        # -----------------------------------------------------
        # ใช้ best.pt ของรอบนี้
        # เป็นโมเดลสำหรับรอบถัดไป
        # -----------------------------------------------------

        current_model = str(best_model)

        print("\nโมเดลที่จะใช้ในรอบถัดไป:")
        print(current_model)

    # =========================================================
    # สรุป
    # =========================================================

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    print(f"จำนวนรอบทั้งหมด : {TOTAL_ROUNDS}")
    print(f"Epoch ต่อรอบ    : {EPOCHS_PER_ROUND}")
    print(f"Epoch สูงสุดรวม : {TOTAL_ROUNDS * EPOCHS_PER_ROUND}")

    print("\nFinal best model:")
    print(current_model)


if __name__ == "__main__":
    main()