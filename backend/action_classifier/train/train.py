import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import csv

# โหลดองค์ประกอบจากพาร์ทที่ 2
from dataset_model import SPHARGruDataset, ActionGRU


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "gru_dataset"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# โฟลเดอร์เก็บผลลัพธ์
RESULT_DIR = Path("training_results")
RESULT_DIR.mkdir(exist_ok=True)

# ไฟล์บันทึกผลการเทรน
CSV_PATH = RESULT_DIR / "training_history.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 กำลังใช้ฮาร์ดแวร์ประมวลผล: {device}")


# ============================================================
# 1. LOAD DATASET
# ============================================================

full_dataset = SPHARGruDataset(DATA_DIR)

num_classes = len(full_dataset.classes)

print(f"📊 ตรวจพบจำนวนคลาสทั้งหมด: {num_classes} คลาส")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 2. MODEL / LOSS / OPTIMIZER
# ============================================================

model = ActionGRU(
    input_size=34,
    hidden_size=64,
    num_layers=2,
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_loss = float("inf")


# ============================================================
# 3. CREATE CSV FILE
# ============================================================

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Train Loss",
        "Validation Loss",
        "Validation Accuracy (%)"
    ])


# ============================================================
# 4. TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for sequences, labels in train_loader:

        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(sequences)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * sequences.size(0)

    train_loss /= len(train_loader.dataset)


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    correct = 0

    with torch.no_grad():

        for sequences, labels in val_loader:

            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * sequences.size(0)

            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels.data)

    val_loss /= len(val_loader.dataset)

    val_acc = (
        correct.double() /
        len(val_loader.dataset)
    ) * 100


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Epoch {epoch+1}/{EPOCHS} -> "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )


    # --------------------------------------------------------
    # Save result to CSV
    # --------------------------------------------------------

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as file:

        writer = csv.writer(file)

        writer.writerow([
            epoch + 1,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{val_acc:.2f}"
        ])


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            RESULT_DIR / "best_action_gru.pth"
        )

        print(
            "   💾 พบค่าน้ำหนักที่ดีกว่าเดิม "
            "บันทึกโมเดลไว้ที่ -> "
            f"{RESULT_DIR / 'best_action_gru.pth'}"
        )


# ============================================================
# FINISH
# ============================================================

print("\n🎉 สิ้นสุดการเทรนโมเดลสำเร็จ!")

print(f"📁 ผลการเทรนอยู่ที่: {CSV_PATH}")
print(
    f"📁 โมเดลที่ดีที่สุดอยู่ที่: "
    f"{RESULT_DIR / 'best_action_gru.pth'}"
)