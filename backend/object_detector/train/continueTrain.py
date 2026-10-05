from ultralytics import YOLO
from pathlib import Path
import shutil
import csv


# ============================================================
# ตั้งค่า
# ============================================================

DATA_YAML = r"dataset/data.yaml"

MODEL = "yolo11n.pt"

TOTAL_EPOCHS = 200
ROUND_SIZE = 50

IMGSZ = 640
BATCH = -1
WORKERS = 4
DEVICE = 0

# โฟลเดอร์หลัก
PROJECT = Path("training_results")

# Run หลักที่ train ต่อเนื่อง
RUN_NAME = "continuous-200"

# ============================================================
# ฟังก์ชันบันทึกผลทุก 50 epochs
# ============================================================

def save_round_checkpoint(trainer):
    """
    ทำงานหลัง train + validation ของแต่ละ epoch

    ทุก ๆ 50 epochs:
    - copy best.pt
    - copy last.pt
    - copy results.csv เฉพาะข้อมูลถึง epoch ปัจจุบัน
    - บันทึก summary.txt
    """

    current_epoch = trainer.epoch + 1

    # ทำงานเฉพาะ epoch 50, 100, 150, 200, 250, 300
    if current_epoch % ROUND_SIZE != 0:
        return

    round_number = current_epoch // ROUND_SIZE

    # --------------------------------------------------------
    # โฟลเดอร์ของรอบ
    # --------------------------------------------------------

    round_dir = (
        PROJECT
        / "checkpoints"
        / f"round-{round_number}"
    )

    round_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Path ของ model หลัก
    # --------------------------------------------------------

    save_dir = Path(trainer.save_dir)

    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    # --------------------------------------------------------
    # Copy best.pt
    # --------------------------------------------------------

    if best_model.exists():
        shutil.copy2(
            best_model,
            round_dir / "best.pt"
        )

    # --------------------------------------------------------
    # Copy last.pt
    # --------------------------------------------------------

    if last_model.exists():
        shutil.copy2(
            last_model,
            round_dir / "last.pt"
        )

    # --------------------------------------------------------
    # Copy results.csv เฉพาะข้อมูลถึง epoch นี้
    # --------------------------------------------------------

    results_csv = save_dir / "results.csv"
    round_csv = round_dir / "results.csv"

    if results_csv.exists():

        try:
            with open(
                results_csv,
                "r",
                encoding="utf-8",
                newline=""
            ) as f:

                reader = list(csv.reader(f))

            if reader:

                header = reader[0]

                # เอาเฉพาะ epoch <= current_epoch
                rows = []

                for row in reader[1:]:

                    if not row:
                        continue

                    try:
                        epoch_value = int(float(row[0]))
                    except (ValueError, IndexError):
                        continue

                    if epoch_value <= current_epoch:
                        rows.append(row)

                with open(
                    round_csv,
                    "w",
                    encoding="utf-8",
                    newline=""
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow(header)
                    writer.writerows(rows)

        except Exception as e:

            print(
                f"[WARNING] ไม่สามารถ copy results.csv: {e}"
            )

    # --------------------------------------------------------
    # บันทึก summary
    # --------------------------------------------------------

    summary_file = round_dir / "summary.txt"

    try:

        metrics = trainer.metrics

        with open(
            summary_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("=" * 60 + "\n")
            f.write(
                f"ROUND {round_number} COMPLETED\n"
            )
            f.write("=" * 60 + "\n\n")

            f.write(
                f"Epoch: {current_epoch}\n\n"
            )

            # Metrics ที่มีจาก validation
            metric_map = {
                "Precision":
                    "metrics/precision(B)",
                "Recall":
                    "metrics/recall(B)",
                "mAP50":
                    "metrics/mAP50(B)",
                "mAP50-95":
                    "metrics/mAP50-95(B)",
            }

            for display_name, key in metric_map.items():

                value = metrics.get(key)

                if value is not None:

                    f.write(
                        f"{display_name}: "
                        f"{value:.6f}\n"
                    )

            f.write("\n")
            f.write(
                f"Best model: {round_dir / 'best.pt'}\n"
            )

            f.write(
                f"Last model: {round_dir / 'last.pt'}\n"
            )

        print("\n" + "=" * 70)
        print(
            f"SAVED ROUND {round_number}"
        )
        print(
            f"Epoch: {current_epoch}"
        )
        print(
            f"Folder: {round_dir}"
        )
        print("=" * 70)

    except Exception as e:

        print(
            f"[WARNING] ไม่สามารถสร้าง summary.txt: {e}"
        )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # สร้างโฟลเดอร์
    # --------------------------------------------------------

    PROJECT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # โหลดโมเดล
    # --------------------------------------------------------

    model = YOLO(MODEL)

    # --------------------------------------------------------
    # เพิ่ม callback
    # --------------------------------------------------------

    model.add_callback(
        "on_fit_epoch_end",
        save_round_checkpoint
    )

    # --------------------------------------------------------
    # Train ต่อเนื่อง 300 epochs
    # --------------------------------------------------------

    print("=" * 70)
    print("START CONTINUOUS TRAINING")
    print("=" * 70)

    print(f"Model          : {MODEL}")
    print(f"Total epochs   : {TOTAL_EPOCHS}")
    print(f"Round size     : {ROUND_SIZE}")
    print(f"Number rounds  : {TOTAL_EPOCHS // ROUND_SIZE}")
    print(f"Image size     : {IMGSZ}")
    print(f"Batch          : {BATCH}")
    print(f"Workers        : {WORKERS}")
    print(f"Device         : {DEVICE}")

    print("=" * 70)

    model.train(

        data=DATA_YAML,

        # ----------------------------------------------------
        # สำคัญ:
        # Train ต่อเนื่อง 300 epochs
        # ----------------------------------------------------

        epochs=TOTAL_EPOCHS,

        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,

        # ----------------------------------------------------
        # ที่เก็บผลหลัก
        # ----------------------------------------------------

        project=str(PROJECT),
        name=RUN_NAME,

        # ----------------------------------------------------
        # บันทึก checkpoint
        # ----------------------------------------------------

        save=True,

        # บันทึก checkpoint ทุก 50 epochs
        # นอกจาก last.pt / best.pt
        save_period=50,

        # ----------------------------------------------------
        # สร้าง plots
        # ----------------------------------------------------

        plots=True,

        # validation
        val=True,

        # ----------------------------------------------------
        # Early stopping
        #
        # 0 = ปิด early stopping
        # เพื่อให้ครบ 300 epochs
        # ----------------------------------------------------

        patience=0,

        verbose=True,
    )

    # ========================================================
    # สรุป
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Train ทั้งหมด: {TOTAL_EPOCHS} epochs"
    )

    print(
        f"แบ่งเก็บผล: ทุก {ROUND_SIZE} epochs"
    )

    print(
        f"\nMain result:"
        f"\n{PROJECT / RUN_NAME}"
    )

    print(
        f"\nRound checkpoints:"
        f"\n{PROJECT / 'checkpoints'}"
    )


if __name__ == "__main__":
    main()