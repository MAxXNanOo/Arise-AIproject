# README.md

## Detect Location for Violent Incidents in Real Time

**Version:** v1.0
**Created:** October 2026
**Project:** Arise
**Institution:** Kasetsart University

---

## 1. Dataset Description

This project uses three datasets to train and evaluate the AI model:

1. **Sound:** Contains normal and abnormal sounds, such as gunshots and screams, to detect abnormal events.

   * *Number of sounds:* 6,000
   * *Number of classes:* 3
   * *Sound format:* WAV
   * *Sound duration:* 1–2 seconds
   * *Sample rate:* 16 kHz

2. **Image:** Contains images of weapons and normal objects to detect weapons during an event.

   * *Number of images:* 16,637
   * *Number of classes:* 5
   * *Image format:* JPG, PNG
   * *Image size:* 640 × 640 pixels

3. **Video:** Contains videos of human actions used to detect dangerous or abnormal events.

   * *Number of videos:* 800
   * *Number of classes:* 6
   * *Video format:* AVI, MP4
   * *Video duration:* 10 seconds
   * *FPS:* 30 fps

---

## 2. Class Labels

1. **Sound Classes**
   | Label | Description |
   | ----- | ----------- |
   | 0     | Gunshot     |
   | 1     | Normal      |
   | 2     | Scream      |

2. **Image Classes**
   | Label |  Description |
   | ----- | ------------ |
   | 0     | Handgun      |
   | 1     | Knife        |
   | 2     | Phone        |
   | 3     | Person       |
   | 4     | Water Bottle |

3. **Video Classes**
   | Label |  Description |
   | ----- | ------------ |
   | 0     | Boxing       |
   | 1     | Handclapping |
   | 2     | Handwaving   |
   | 3     | Jogging      |
   | 4     | Running      |
   | 5     | Walking      |

---

## 3. Dataset Structure

```text
Dataset/
├── Sound/
│   ├── 0_Gunshot/
│   ├── 1_Normal/
│   └── 2_Scream/
│
├── Image/
│   ├── 0_Handgun/
│   ├── 1_Knife/
│   ├── 2_Phone/
│   ├── 3_Person/
│   └── 4_WaterBottle/
│
└── Video/
    ├── 0_Boxing/
    ├── 1_Handclapping/
    ├── 2_Handwaving/
    ├── 3_Jogging/
    ├── 4_Running/
    └── 5_Walking/
```

---

## 4. Data Source

All datasets were collected from publicly available open-source datasets online.

* **Sound Dataset:**

  * [Human Screaming Detection Dataset](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset)
  * [Gunshot Audio Dataset](https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset?select=M249)

* **Image Dataset:**

  * [Handgun Dataset](https://huggingface.co/datasets/JoseArmando07/gun-dataset)
  * [Phone Dataset](https://universe.roboflow.com/z-jeans-pig/phone-oldo4)
  * [Water Bottle Dataset](https://universe.roboflow.com/waterbottles/water-bottles-0dgex)
  * [Person Dataset](https://universe.roboflow.com/keke/person-fzqgw/dataset/12)
  * [Knife Detection Dataset](https://universe.roboflow.com/ai-0jtbr/knife-detection-hgvy2)

* **Video Dataset:**

  * [KTH Action Recognition Dataset](https://www.kaggle.com/datasets/vafaeii/kth-action-recognition-dataset/data)

---

## 5. Annotation

The datasets were categorized into classes based on the type of sound, object, and human action.

* **Sound:** Audio files were categorized into three classes based on the detected sound events. **Gunshot** and **Scream** represent potential signs of violent incidents, while **Normal** represents non-violent sounds.
* **Image:** Images were categorized into five classes based on the detected objects. **Handgun** and **Knife** represent potential weapons related to violent incidents, while **Phone, Person,** and **Water Bottle** represent normal objects.
* **Video:** Videos were categorized into six classes based on human actions. **Boxing** represents a potential violent action, while **Handclapping, Handwaving, Jogging, Running,** and **Walking** represent normal human activities.

---

## 6. Recommended Data Split

1. **Sound Classes**
   | Subset     | Percentage |
   | ---------- | ---------: |
   | Training   |        70% |
   | Validation |        15% |
   | Testing    |        15% |

2. **Image Classes**
   | Subset     | Percentage |
   | ---------- | ---------: |
   | Training   |        60% |
   | Validation |        30% |
   | Testing    |        10% |

3. **Video Classes**
   | Subset     | Percentage |
   | ---------- | ---------: |
   | Training   |        80% |
   | Validation |        20% |
   | Testing    |        0   |
   
---

## 7. Version History

### Sound Classes

| Version | Description                           |
| ------- | ------------------------------------- |
| v1.0    | Initial version with too many classes |
| v2.0    | Reduced to 3 classes                  |
| v3.0    | Added more normal sounds              |

### Image Classes

| Version | Description                                          |
| ------- | ---------------------------------------------------- |
| v1.0    | Initial version with detection errors                |
| v2.0    | Added more handgun images                            |
| v3.0    | Added Knife, Phone, Water Bottle, and Person classes |

### Video Classes

| Version | Description                                                 |
| ------- | ----------------------------------------------------------- |
| v1.0    | Initial version where all videos were classified as Sitting |
| v2.0    | Removed the Sitting class                                   |
 
 ---

## 8.System overview

```
                ┌─────────────────────┐
  Camera ─────► │ YOLO Object Detect  │──► objects ──┐
     │          └─────────────────────┘              │
     │          ┌─────────────────────┐              │
     └────────► │ YOLO-Pose (keypoints)│             │
                └──────────┬──────────┘              ▼
                           ▼                   ┌──────────┐     ┌──────────┐
                  Sequence buffer (T frames)   │  Fusion  │────►│ Web API  │
                           ▼                   │  Logic   │     │ / Socket │
                    ┌────────────┐             └──────────┘     └────┬─────┘
                    │    GRU     │──► action ───────▲                ▼
                    └────────────┘                  │           Web Dashboard
  Mic ────► YAMNet ──► audio event ─────────────────┘
```
### 9.การทำงาน
- **Object Detection**  
  ตรวจจับวัตถุที่เป็นอาวุธจากภาพ เช่น ปืนหรือมีด พร้อมประเมินค่าความมั่นใจของการตรวจจับ

- **Audio Event Classification**  
  วิเคราะห์เสียงจากกล้อง เพื่อตรวจจับเสียงที่เกี่ยวข้องกับเหตุการณ์ เช่น **เสียงปืน (Gunshot/Gunfire)** และ **เสียงกรีดร้อง (Screaming)**

- **Action Recognition**  
  ใช้ **YOLO-Pose** ในการดึง Keypoints ของร่างกายจากภาพ และส่งข้อมูลต่อให้ **GRU** วิเคราะห์ลำดับการเคลื่อนไหว เพื่อจำแนกพฤติกรรมของบุคคล เช่น การวิ่ง การเดิน หรือการต่อสู้

- **Logic** ใช้การตรวจความเเม่นยำถ้าผ่านเกณจะไปเช็คตามสถานการที่เรากำหนดไว้ถ้าผ่านส่งให้กับเว็ป

- **Web Integration**  
  นำผลลัพธ์จาก AI ทั้ง 3 ส่วนมาประมวลผลร่วมกันผ่าน **Logic และ Event Correlation** เพื่อคำนวณคะแนนความเสี่ยงของเหตุการณ์ และส่งผลลัพธ์ไปยังหน้าเว็บไซต์แบบเรียลไทม์



## 10.โครงสร้างโฟลเดอร์
```
Arise/
├── README.md
├── requirements.txt
├── configAI.json
├── camera.json
├── main.py                             # โค้ดหลักในการทำงาน
├── backend/                            # โค้ดการทำงานของเเต่ละตัว เเละโค้ดการเทรนที่ใช้
│   ├── audio_classifier/
|   |   ├── audio_classifier.py
|   |   └── train/
│   ├── object_detector/
|   |   ├── object_detector.py
|   |   └── train/
│   └── action_classifier/
|       ├── action_classifier.py
|       └── train/
├── models/                             # เก็บ model ที่เราเทรนไว้
│   ├── audio_classifier/
│   ├── object_detector/
│   └── action_classifier/
├── training/
│   ├── prepare_dataset.py
│   └── train_gru.py
|
├── index.html                          # Web
├── script.js
└── api_server.py
```


## 11.หน้าที่งานเเต่ละไฟล์

- **`configAI.json`**  
  เก็บค่าการตั้งค่าของระบบ เช่น เงื่อนไขการแจ้งเตือน, threshold, PATH ของ AI และ URL ของ Web API
  - `event_correlation` ─ ไว้เก็บเงื่อนไขตามสถานการต่างๆที่เราอยากตรวจสอบ
  - `models` ─ ตั้งค่า PATH เเละ Class ของตัว AI
- **`camera.json`**  
  เก็บข้อมูลกล้อง เช่น ชื่อกล้อง ตำแหน่ง อาคาร ชั้น และพิกัด Latitude/Longitude เพื่อใช้ระบุตำแหน่งเมื่อเกิดเหตุการณ์

- **`main.py`**  
  ไฟล์หลักของระบบ ทำหน้าที่ควบคุมการทำงานของ AI ตั้งแต่รับข้อมูลจากกล้องและไมโครโฟน ประมวลผล ตรวจจับเหตุการณ์ และส่งข้อมูลไปยัง Web API
  - `ObjectDetecter` ─ ใช้การหาสิ่งของเช่นพวกอาวุธ
  - `PoseEstimator + ActionClassifier` ─ ใช้ในการตรวจจับท่าทาง
  - `AudioController` ─ ใช้ในการตรวจจับเสียงว่าได้ยินเสียงอะไร
  - `check_*_alert` ─ ใช้ตรวจว่าค่าความมั่นใจที่ได้มามันมากพอที่่จะเอามารึป่าว
  - `EventCorrelationEngine` ─ ตรวจค่าตาม configAI ที่วางไว้
  - `AlertSender` ─ ถ้ามันผ่านจากการตรวจก่อนหน้าจะทำการส่งขึ้นเว็ป

- **`backend/object_detector/object_detector.py`**  
  ทำหน้าที่ตรวจจับวัตถุจากภาพ เช่น ปืน มีด และวัตถุอื่น ๆ รอ main.py เรียกใช้

- **`backend/action_classifier/pose_estimator.py`**  
  ตรวจจับตำแหน่ง Keypoints ของบุคคลจากภาพด้วย YOLO-Pose รอ main.py เรียกใช้

- **`backend/action_classifier/action_classifier.py`**  
  นำ Keypoints จากหลายเฟรมเข้าสู่ GRU เพื่อจำแนกการกระทำของบุคคล รอ main.py เรียกใช้

- **`backend/audio_classifier/audio_classifier.py`**  
  วิเคราะห์เสียงจากไมโครโฟนและจำแนกประเภทของเสียง เช่น เสียงปืนและเสียงกรีดร้อง รอ main.py เรียกใช้

- **`api_server.py`**  
  ทำหน้าที่เป็น Web API สำหรับรับข้อมูลจากระบบ AI และส่งข้อมูลไปแสดงผลบนเว็บไซต์

- **`index.html`**  
  หน้าเว็บไซต์สำหรับแสดงข้อมูลและสถานะของระบบ

- **`script.js`**  
  ควบคุมการทำงานของหน้าเว็บไซต์และรับข้อมูลจาก Web API แบบ Real-time





# Installation - Windows

## Create Virtual Environment
    
```powershell
python -m venv venv
```

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
```powershell
pip install -r requirements.txt
```
Or :

```powershell
pip install numpy
pip install torch
pip install torchvision
pip install opencv-python
pip install ultralytics
pip install fastapi
pip install uvicorn
pip install requests
pip install sounddevice
pip install tensorflow
pip install tensorflow-hub
pip install scikit-learn
```

---

# Installation - Linux / Ubuntu

## Update System

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y portaudio19-dev
sudo apt install -y build-essential
```

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

หรือ:

```bash
pip install numpy
pip install torch
pip install torchvision
pip install opencv-python
pip install ultralytics
pip install fastapi
pip install uvicorn
pip install requests
pip install sounddevice
pip install tensorflow
pip install tensorflow-hub
pip install scikit-learn
```


## วิธีใช้งาน

```bash
# รันไฟล์หลัก
python main.py
# รันไฟล์สำหรับเซอร์เวอร์เชื่อมหน้าเว็ป
python api_server.py
```
- เข้าเว็ป http://127.0.0.1:5502/index.html


## การส่งข้อมูลไปที่เว็บ

### ตัวอย่าง JSON ส่งให้เว็ป

```json
{
  "timestamp": "2026-10-05T14:23:48.640873",
  "camera": {
    "id": "1",
    "name": "Camera 1",
    "latitude": 14.020792,
    "longitude": 99.971371,
    "location": "ห้องเรียน AI",
    "building": "E9",
    "floor": "ชั้น 4",
    "description": "ห้องเรียน"
  },
  "object_detection": [
    {
      "class_id": 2,
      "class_name": "phone",
      "confidence": 0.8292937278747559,
      "bbox": [
        0,
        126,
        220,
        577
      ]
    }
  ],
  "action_gru": [
    {
      "track_id": 357,
      "class_id": 5,
      "class_name": "walking",
      "confidence": 0.7826719880104065
    },
    {
      "track_id": 362,
      "class_id": 5,
      "class_name": "walking",
      "confidence": 0.9991672039031982
    }
  ],
  "audio_yamnet": {
    "class_name": "1",
    "confidence": 0.9999979734420776
  }
}
```



# Contact

**Arise Project team**

Department of Computer Engineering  
Faculty of Engineering at Kamphaeng Saen  
Kasetsart University, Thailand

**Project member**
|             Name              |       Email        |
|-------------------------------|--------------------|
| Khullakan Saharattanachot     | khullakan.s@ku.th  |
| Nattawat Taweerojpinyo        | Nattawat.taw@ku.th |
| thanakit wanmee               | thanakit.wan@ku.th |
| Sitharinee Suranan            | sitharinee.s@ku.th |