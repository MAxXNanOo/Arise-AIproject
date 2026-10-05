import os
import json
import time
import uuid
import base64
import threading
import multiprocessing as mp
from datetime import datetime

import cv2
import requests

from backend.object_detector.object_detector import ObjectDetector
from backend.action_classifier.pose_estimator import PoseEstimator
from backend.action_classifier.action_classifier import ActionClassifier
from backend.audio_classifier.audio_classifier import AudioController


print("=" * 70)
print("ARISE AI - MAIN CONTROLLER")
print("=" * 70)


# ---------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "configAI.json"
)

CAMERA_CONFIG_PATH = os.path.join(
    BASE_DIR,
    "camera.json"
)


# ---------------------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------------------

def load_json(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


config = load_json(CONFIG_PATH)

camera_config = load_json(
    CAMERA_CONFIG_PATH
)


# ---------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------

def now_iso():

    return datetime.now().isoformat()


def frame_to_base64(frame):

    if frame is None:
        return None

    try:

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                80
            ]
        )

        if not success:
            return None

        return base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8")

    except Exception as e:

        print(
            f"[Base64] Error: {e}"
        )

        return None


def create_incident_id():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    random_part = uuid.uuid4().hex[:6]

    return (
        f"{timestamp}_{random_part}"
    )


# ---------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------

def select_camera(camera_data):

    print("=" * 70)
    print("ARISE CAMERA SELECTION")
    print("=" * 70)

    for camera_id, camera in camera_data.items():

        print(
            f"[{camera_id}] "
            f"{camera.get('name', 'Unknown')} | "
            f"{camera.get('location', '-')} | "
            f"{camera.get('building', '-')} | "
            f"{camera.get('floor', '-')}"
        )

    print("=" * 70)

    while True:

        selected = input(
            "เลือกหมายเลขกล้อง: "
        ).strip()

        if selected in camera_data:
            break

        print(
            "[ERROR] ไม่พบกล้องหมายเลขนี้"
        )

    camera = camera_data[selected]

    camera_info = {

        "id":
            str(selected),

        "name":
            camera.get(
                "name",
                f"Camera {selected}"
            ),

        "latitude":
            camera.get(
                "latitude",
                0.0
            ),

        "longitude":
            camera.get(
                "longitude",
                0.0
            ),

        "location":
            camera.get(
                "location",
                ""
            ),

        "building":
            camera.get(
                "building",
                ""
            ),

        "floor":
            camera.get(
                "floor",
                ""
            ),

        "description":
            camera.get(
                "description",
                ""
            )
    }

    print()
    print("-" * 70)
    print("Selected Camera")
    print("-" * 70)

    print(
        f"ID          : {camera_info['id']}"
    )

    print(
        f"Name        : {camera_info['name']}"
    )

    print(
        f"Location    : {camera_info['location']}"
    )

    print(
        f"Building    : {camera_info['building']}"
    )

    print(
        f"Floor       : {camera_info['floor']}"
    )

    print(
        f"Description : {camera_info['description']}"
    )

    print(
        f"Latitude    : {camera_info['latitude']}"
    )

    print(
        f"Longitude   : {camera_info['longitude']}"
    )

    print("-" * 70)
    print()

    return camera_info


# ---------------------------------------------------------------------
# ALERT SENDER
# ---------------------------------------------------------------------

class AlertSender:

    def __init__(
        self,
        config,
        camera_info
    ):

        alert_config = config.get(
            "web_alert",
            {}
        )

        self.enabled = alert_config.get(
            "enabled",
            True
        )

        self.url = alert_config.get(
            "url",
            "http://localhost:5500/api/alerts"
        )

        self.timeout = float(
            alert_config.get(
                "timeout_sec",
                2.0
            )
        )

        self.camera = camera_info

    def send_incident(
        self,
        event_type,
        class_name,
        confidence,
        frame,
        extra_data=None
    ):

        if not self.enabled:
            return

        incident_id = create_incident_id()

        image_base64 = frame_to_base64(
            frame
        )

        payload = {

            "incident_id":
                incident_id,

            "timestamp":
                now_iso(),

            "status":
                "active",

            "event_type":
                event_type,

            "class_name":
                class_name,

            "confidence":
                float(confidence),

            "camera":
                self.camera,

            "image":
                image_base64,

            "data":
                extra_data or {}
        }

        def send():

            try:

                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout
                )

                print()
                print("=" * 70)
                print("🚨 INCIDENT SENT")
                print("=" * 70)

                print(
                    f"Incident ID : {incident_id}"
                )

                print(
                    f"Type        : {event_type}"
                )

                print(
                    f"Class       : {class_name}"
                )

                print(
                    f"Camera      : "
                    f"{self.camera['id']}"
                )

                print(
                    f"HTTP        : "
                    f"{response.status_code}"
                )

                print("=" * 70)

            except Exception as e:

                print(
                    f"[AlertSender] Failed: {e}"
                )

        threading.Thread(
            target=send,
            daemon=True
        ).start()


# ---------------------------------------------------------------------
# ALERT CHECK
#
# ตรงนี้เป็นคนตรวจ threshold ของ AI แต่ละตัว
# ---------------------------------------------------------------------

def check_object_alert(
    detections,
    config
):

    alerts = []

    object_config = config[
        "models"
    ][
        "object_detector"
    ][
        "classes"
    ]

    if not detections:
        return alerts

    for detection in detections:

        if not isinstance(
            detection,
            dict
        ):
            continue

        class_id = detection.get(
            "class_id"
        )

        class_name = detection.get(
            "class_name",
            "unknown"
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0
            )
        )

        class_cfg = object_config.get(
            str(class_id)
        )

        if class_cfg is None:
            continue

        threshold = float(
            class_cfg.get(
                "threshold",
                0.0
            )
        )

        if not class_cfg.get(
            "alert",
            False
        ):
            continue

        if threshold <= 0:
            continue

        if confidence >= threshold:

            alerts.append({

                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "confidence":
                    confidence,

                "bbox":
                    detection.get(
                        "bbox"
                    )
            })

    return alerts


def check_action_alert(
    action_results,
    config
):

    alerts = []

    if not action_results:
        return alerts

    action_config = config[
        "models"
    ][
        "action_classifier"
    ][
        "classes"
    ]

    for action in action_results:

        if not isinstance(
            action,
            dict
        ):
            continue

        class_id = action.get(
            "class_id"
        )

        class_name = action.get(
            "class_name",
            "unknown"
        )

        confidence = float(
            action.get(
                "confidence",
                0.0
            )
        )

        track_id = action.get(
            "track_id"
        )

        class_cfg = action_config.get(
            str(class_id)
        )

        if class_cfg is None:
            continue

        threshold = float(
            class_cfg.get(
                "threshold",
                0.0
            )
        )

        if not class_cfg.get(
            "alert",
            False
        ):
            continue

        if threshold <= 0:
            continue

        if confidence >= threshold:

            alerts.append({

                "track_id":
                    track_id,

                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "confidence":
                    confidence
            })

    return alerts


def check_audio_alert(
    audio,
    config
):

    alerts = []

    if not isinstance(
        audio,
        dict
    ):
        return alerts

    class_name = audio.get(
        "class_name",
        "No Audio"
    )

    confidence = float(
        audio.get(
            "confidence",
            0.0
        )
    )

    audio_config = config[
        "models"
    ][
        "audio_classifier"
    ][
        "classes"
    ]

    class_cfg = audio_config.get(
        class_name
    )

    if class_cfg is None:
        return alerts

    alert_enabled = class_cfg.get(
        "alert",
        False
    )

    threshold = float(
        class_cfg.get(
            "threshold",
            0.0
        )
    )

    if not alert_enabled:
        return alerts

    if threshold <= 0:
        return alerts

    if confidence >= threshold:

        alerts.append({

            "class_name":
                class_name,

            "confidence":
                confidence
        })

    return alerts


# ---------------------------------------------------------------------
# EVENT CORRELATION
#
# ไม่มี score
# ไม่มี weight
# ไม่มี min_score
# ไม่มีการคำนวณ
#
# อ่านเงื่อนไขจาก configAI.json
# แล้วใช้ IF / ELSE อย่างเดียว
# ---------------------------------------------------------------------

class EventCorrelationEngine:

    VALID_SOURCES = (
        "action",
        "object",
        "audio",
        "people"
    )

    def __init__(
        self,
        config
    ):

        correlation_config = config.get(
            "event_correlation",
            {}
        )

        self.enabled = correlation_config.get(
            "enabled",
            True
        )

        self.rules = correlation_config.get(
            "rules",
            {}
        )

        self.last_incident_time = {}

        self.validate_rules()

    # -------------------------------------------------------------
    # VALIDATE CONFIG
    # -------------------------------------------------------------

    def validate_rules(self):

        for rule_name, rule in self.rules.items():

            signals = rule.get(
                "signals",
                {}
            )

            for signal_name, signal in signals.items():

                source = signal.get(
                    "source"
                )

                if source not in self.VALID_SOURCES:

                    print(
                        "[Correlation] "
                        f"WARNING: rule "
                        f"'{rule_name}' "
                        f"signal '{signal_name}' "
                        f"source ไม่ถูกต้อง: "
                        f"{source}"
                    )

    # -------------------------------------------------------------
    # CHECK COOLDOWN
    # -------------------------------------------------------------

    def is_in_cooldown(
        self,
        rule_name,
        rule
    ):

        cooldown = float(
            rule.get(
                "cooldown_sec",
                10
            )
        )

        current_time = time.time()

        last_time = self.last_incident_time.get(
            rule_name,
            0
        )

        if current_time - last_time < cooldown:

            return True

        return False

    # -------------------------------------------------------------
    # GET SIGNAL
    # -------------------------------------------------------------

    @staticmethod
    def has_signal(
        alerts,
        classes
    ):

        if not alerts:
            return False, []

        matched = []

        for alert in alerts:

            class_name = alert.get(
                "class_name"
            )

            if class_name in classes:

                matched.append(
                    alert
                )

        if matched:

            return True, matched

        return False, []

    # -------------------------------------------------------------
    # COUNT PEOPLE
    # -------------------------------------------------------------

    @staticmethod
    def count_people(
        action_alerts,
        action_classes
    ):

        people = {}

        for index, alert in enumerate(
            action_alerts
        ):

            class_name = alert.get(
                "class_name"
            )

            if class_name not in action_classes:
                continue

            track_id = alert.get(
                "track_id"
            )

            if track_id is None:

                track_id = (
                    f"person_{index}"
                )

            if track_id not in people:

                people[track_id] = alert

        return list(
            people.values()
        )

    # -------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------

    def evaluate(
        self,
        object_alerts,
        action_alerts,
        audio_alerts
    ):

        if not self.enabled:

            return None

        # ---------------------------------------------------------
        # วนตาม rule ใน config
        # ---------------------------------------------------------

        for rule_name, rule in self.rules.items():

            if not rule.get(
                "enabled",
                True
            ):

                continue

            # -----------------------------------------------------
            # Cooldown
            # -----------------------------------------------------

            if self.is_in_cooldown(
                rule_name,
                rule
            ):

                continue

            signals = rule.get(
                "signals",
                {}
            )

            # -----------------------------------------------------
            # ตัวแปรสถานะ
            # -----------------------------------------------------

            action_ok = True
            object_ok = True
            audio_ok = True
            people_ok = True

            matched_actions = []
            matched_objects = []
            matched_audio = []
            matched_people = []

            # =====================================================
            # ACTION
            # =====================================================

            if "action" in signals:

                signal = signals[
                    "action"
                ]

                classes = signal.get(
                    "classes",
                    []
                )

                action_ok, matched_actions = (
                    self.has_signal(
                        action_alerts,
                        classes
                    )
                )

            # =====================================================
            # OBJECT
            # =====================================================

            if "object" in signals:

                signal = signals[
                    "object"
                ]

                classes = signal.get(
                    "classes",
                    []
                )

                object_ok, matched_objects = (
                    self.has_signal(
                        object_alerts,
                        classes
                    )
                )

            # =====================================================
            # AUDIO
            # =====================================================

            if "audio" in signals:

                signal = signals[
                    "audio"
                ]

                classes = signal.get(
                    "classes",
                    []
                )

                audio_ok, matched_audio = (
                    self.has_signal(
                        audio_alerts,
                        classes
                    )
                )

            # =====================================================
            # PEOPLE
            # =====================================================

            if "people" in signals:

                signal = signals[
                    "people"
                ]

                min_people = int(
                    signal.get(
                        "min_people",
                        1
                    )
                )

                # คนที่นับจะอิงจาก action
                # ของ rule นี้

                action_signal = signals.get(
                    "action",
                    {}
                )

                action_classes = action_signal.get(
                    "classes",
                    []
                )

                matched_people = (
                    self.count_people(
                        action_alerts,
                        action_classes
                    )
                )

                if len(
                    matched_people
                ) >= min_people:

                    people_ok = True

                else:

                    people_ok = False

            # =====================================================
            # CHECK ALL CONDITIONS
            # =====================================================

            if (
                action_ok
                and object_ok
                and audio_ok
                and people_ok
            ):

                # ผ่านเงื่อนไขทั้งหมด

                current_time = time.time()

                self.last_incident_time[
                    rule_name
                ] = current_time

                evidence = []

                evidence.extend(
                    matched_actions
                )

                evidence.extend(
                    matched_objects
                )

                evidence.extend(
                    matched_audio
                )

                return {

                    "type":
                        rule_name,

                    "class_name":
                        rule_name,

                    "confidence":
                        1.0,

                    "people":
                        matched_people,

                    "actions":
                        matched_actions,

                    "objects":
                        matched_objects,

                    "audio":
                        matched_audio,

                    "evidence":
                        evidence,

                    "reason":
                        f"Rule '{rule_name}' "
                        "ผ่านเงื่อนไขทั้งหมด"
                }

            # -----------------------------------------------------
            # ไม่ผ่าน
            # -----------------------------------------------------

            else:

                continue

        return None


# ---------------------------------------------------------------------
# DRAW
# ---------------------------------------------------------------------

def draw_camera_info(
    frame,
    camera_info
):

    y = 30

    lines = [

        f"ARISE AI | Camera "
        f"{camera_info['id']}",

        f"{camera_info['location']} | "
        f"{camera_info['building']} | "
        f"{camera_info['floor']}"
    ]

    for line in lines:

        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        y += 30


def draw_actions(
    frame,
    actions
):

    y = 100

    for action in actions:

        if not isinstance(
            action,
            dict
        ):
            continue

        track_id = action.get(
            "track_id",
            -1
        )

        class_name = action.get(
            "class_name",
            "unknown"
        )

        confidence = float(
            action.get(
                "confidence",
                0
            )
        )

        text = (
            f"ID {track_id}: "
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        y += 28


def draw_audio(
    frame,
    audio
):

    if not isinstance(
        audio,
        dict
    ):
        return

    class_name = audio.get(
        "class_name",
        "No Audio"
    )

    confidence = float(
        audio.get(
            "confidence",
            0
        )
    )

    text = (
        f"Audio: "
        f"{class_name} "
        f"{confidence:.2f}"
    )

    height = frame.shape[0]

    cv2.putText(
        frame,
        text,
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ---------------------------------------------------------------------
# WEB API
# ---------------------------------------------------------------------

def build_ai_payload(
    camera_info,
    object_results,
    action_results,
    audio_result
):

    return {

        "timestamp":
            now_iso(),

        "camera":
            camera_info,

        "object_detection":
            object_results,

        "action_gru":
            action_results,

        "audio_yamnet":
            audio_result
    }


def send_ai_data(
    payload,
    url
):

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=1.5
        )

        return response

    except Exception:

        return None


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    camera_info = None
    audio_controller = None
    cap = None

    try:

        camera_info = select_camera(
            camera_config
        )

        print(
            "[MAIN] Loading AI modules..."
        )

        # ---------------------------------------------------------
        # AI
        # ---------------------------------------------------------

        object_detector = ObjectDetector(
            config
        )

        pose_estimator = PoseEstimator(
            config
        )

        action_classifier = ActionClassifier(
            config
        )

        audio_controller = AudioController(
            config
        )

        audio_controller.start()

        # ---------------------------------------------------------
        # Alert / Event
        # ---------------------------------------------------------

        alert_sender = AlertSender(
            config,
            camera_info
        )

        event_engine = EventCorrelationEngine(
            config
        )

        # ---------------------------------------------------------
        # Camera Config
        # ---------------------------------------------------------

        camera_settings = config.get(
            "camera",
            {}
        )

        device_index = int(
            camera_settings.get(
                "device_index",
                0
            )
        )

        width = int(
            camera_settings.get(
                "width",
                1280
            )
        )

        height = int(
            camera_settings.get(
                "height",
                720
            )
        )

        fps = int(
            camera_settings.get(
                "fps",
                30
            )
        )

        print()
        print(
            f"[Camera] Device: "
            f"{device_index}"
        )

        print(
            f"[Camera] Resolution: "
            f"{width}x{height}"
        )

        print(
            f"[Camera] FPS: "
            f"{fps}"
        )

        # ---------------------------------------------------------
        # Open Camera
        # ---------------------------------------------------------

        cap = cv2.VideoCapture(
            device_index
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )

        cap.set(
            cv2.CAP_PROP_FPS,
            fps
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"ไม่สามารถเปิด "
                f"Camera {device_index}"
            )

        print()
        print("=" * 70)
        print("✓ ARISE AI STARTED")
        print("=" * 70)

        print(
            f"Camera ID : "
            f"{camera_info['id']}"
        )

        print(
            f"Location  : "
            f"{camera_info['location']}"
        )

        print(
            f"Building  : "
            f"{camera_info['building']}"
        )

        print(
            f"Floor     : "
            f"{camera_info['floor']}"
        )

        print()
        print("กด Q เพื่อออก")
        print("=" * 70)

        # ---------------------------------------------------------
        # Web API
        # ---------------------------------------------------------

        ai_api_url = config.get(
            "web_api",
            {}
        ).get(
            "ai_url",
            "http://localhost:5500/api/ai"
        )

        last_json_time = 0

        # ---------------------------------------------------------
        # LOOP
        # ---------------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "[Camera] "
                    "ไม่สามารถอ่าน frame"
                )

                break

            # =====================================================
            # OBJECT DETECTION
            # =====================================================

            object_results, object_frame = (
                object_detector.infer(
                    frame
                )
            )

            object_alerts = (
                check_object_alert(
                    object_results,
                    config
                )
            )

            # =====================================================
            # POSE
            # =====================================================

            pose_results, pose_frame = (
                pose_estimator.infer(
                    object_frame
                )
            )

            # =====================================================
            # ACTION
            # =====================================================

            try:

                action_results = (
                    action_classifier.update(
                        pose_results,
                        frame.shape
                    )
                )

            except Exception as e:

                print(
                    "[ActionClassifier] "
                    f"Update error: {e}"
                )

                action_results = []

            action_alerts = (
                check_action_alert(
                    action_results,
                    config
                )
            )

            # =====================================================
            # AUDIO
            # =====================================================

            audio_result = (
                audio_controller.get_latest()
            )

            audio_alerts = (
                check_audio_alert(
                    audio_result,
                    config
                )
            )

            # =====================================================
            # DRAW
            # =====================================================

            display = pose_frame.copy()

            draw_camera_info(
                display,
                camera_info
            )

            draw_actions(
                display,
                action_results
            )

            draw_audio(
                display,
                audio_result
            )

            # =====================================================
            # EVENT CORRELATION
            # =====================================================

            correlation_result = (
                event_engine.evaluate(
                    object_alerts,
                    action_alerts,
                    audio_alerts
                )
            )

            # =====================================================
            # SEND INCIDENT
            # =====================================================

            if correlation_result is not None:

                event_type = (
                    correlation_result[
                        "type"
                    ]
                )

                people = (
                    correlation_result.get(
                        "people",
                        []
                    )
                )

                evidence = (
                    correlation_result.get(
                        "evidence",
                        []
                    )
                )

                reason = (
                    correlation_result.get(
                        "reason",
                        ""
                    )
                )

                print()
                print("=" * 70)
                print(
                    "🚨 CORRELATED EVENT DETECTED"
                )
                print("=" * 70)

                print(
                    f"Type     : "
                    f"{event_type}"
                )

                print(
                    f"People   : "
                    f"{len(people)}"
                )

                print(
                    f"Evidence : "
                    f"{len(evidence)}"
                )

                print(
                    f"Reason   : "
                    f"{reason}"
                )

                print("=" * 70)

                alert_sender.send_incident(

                    event_type=
                        event_type,

                    class_name=
                        event_type,

                    confidence=
                        1.0,

                    frame=
                        display,

                    extra_data={

                        "people":
                            people,

                        "actions":
                            correlation_result.get(
                                "actions",
                                []
                            ),

                        "objects":
                            correlation_result.get(
                                "objects",
                                []
                            ),

                        "audio":
                            correlation_result.get(
                                "audio",
                                []
                            ),

                        "reason":
                            reason,

                        "evidence":
                            evidence
                    }
                )

            # =====================================================
            # SEND AI DATA
            # =====================================================

            payload = build_ai_payload(

                camera_info=
                    camera_info,

                object_results=
                    object_results,

                action_results=
                    action_results,

                audio_result=
                    audio_result
            )

            current_time = time.time()

            if (
                current_time
                - last_json_time
                >= 1.0
            ):

                print()

                print(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2
                    )
                )

                send_ai_data(
                    payload,
                    ai_api_url
                )

                last_json_time = (
                    current_time
                )

            # =====================================================
            # DISPLAY
            # =====================================================

            cv2.imshow(
                "ARISE AI",
                display
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

    except KeyboardInterrupt:

        print()
        print(
            "[ARISE] Interrupted"
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("[ARISE ERROR]")
        print(str(e))
        print("=" * 70)

        import traceback

        traceback.print_exc()

    finally:

        print()
        print(
            "Cleaning up..."
        )

        if cap is not None:

            try:
                cap.release()

            except Exception:
                pass

        if audio_controller is not None:

            try:
                audio_controller.stop()

            except Exception:
                pass

        try:

            cv2.destroyAllWindows()

        except Exception:

            pass

        print(
            "ARISE stopped."
        )


# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    mp.freeze_support()

    try:

        mp.set_start_method(
            "spawn",
            force=True
        )

    except RuntimeError:

        pass

    main()