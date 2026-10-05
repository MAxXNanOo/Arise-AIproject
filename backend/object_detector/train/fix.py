from pathlib import Path

# กำหนด path ของ folder
folder_path = Path(r"waterBottle/valid/labels")

# วนทุกไฟล์ .txt ใน folder
for file_path in folder_path.glob("*.txt"):
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()

        new_lines = []

        for line in lines:
            line = line.strip()

            # ถ้าบรรทัดว่าง
            if not line:
                new_lines.append(line)
                continue

            # แยกข้อมูลในบรรทัด
            parts = line.split()

            # ถ้าตัวแรกเป็น 0 ให้เปลี่ยนเป็น 3
            if parts[0] == "0":
                parts[0] = "4"

            # รวมข้อมูลกลับเป็นบรรทัด
            new_lines.append(" ".join(parts))

        # บันทึกทับไฟล์เดิม
        file_path.write_text(
            "\n".join(new_lines),
            encoding="utf-8"
        )

        print(f"แก้ไขแล้ว: {file_path.name}")

    except Exception as e:
        print(f"เกิดข้อผิดพลาดกับ {file_path.name}: {e}")

print("เสร็จสิ้น")