# import queue
# import multiprocessing as mp

# import numpy as np


# def yamnet_worker(
#     input_queue,
#     output_queue,
#     sample_rate,
#     chunk_duration
# ):

#     try:

#         import sounddevice as sd
#         import tensorflow as tf
#         import tensorflow_hub as hub

#         print("[Audio] Loading YAMNet...")

#         yamnet = hub.load(
#             "https://tfhub.dev/google/yamnet/1"
#         )

#         class_map_path = tf.keras.utils.get_file(
#             "yamnet_class_map.csv",
#             "https://raw.githubusercontent.com/"
#             "tensorflow/models/master/"
#             "research/audioset/yamnet/"
#             "yamnet_class_map.csv"
#         )

#         class_names = []

#         with open(
#             class_map_path,
#             "r",
#             encoding="utf-8"
#         ) as f:

#             next(f)

#             for line in f:

#                 parts = line.strip().split(
#                     ",",
#                     2
#                 )

#                 if len(parts) >= 3:
#                     class_names.append(
#                         parts[2].strip()
#                     )

#         print("[Audio] YAMNet ready")

#         block_size = int(
#             sample_rate *
#             chunk_duration
#         )

#         while True:

#             try:

#                 audio = sd.rec(
#                     block_size,
#                     samplerate=sample_rate,
#                     channels=1,
#                     dtype="float32"
#                 )

#                 sd.wait()

#                 audio = audio.reshape(-1)

#                 if len(audio) == 0:

#                     output_queue.put({
#                         "class_name": "No Audio",
#                         "confidence": 0.0
#                     })

#                     continue

#                 scores, embeddings, spectrogram = (
#                     yamnet(
#                         tf.convert_to_tensor(
#                             audio,
#                             dtype=tf.float32
#                         )
#                     )
#                 )

#                 scores_np = scores.numpy()

#                 mean_scores = np.mean(
#                     scores_np,
#                     axis=0
#                 )

#                 class_id = int(
#                     np.argmax(mean_scores)
#                 )

#                 confidence = float(
#                     mean_scores[class_id]
#                 )

#                 if class_id < len(
#                     class_names
#                 ):

#                     class_name = (
#                         class_names[class_id]
#                     )

#                 else:

#                     class_name = str(
#                         class_id
#                     )

#                 output_queue.put({

#                     "class_name":
#                         class_name,

#                     "confidence":
#                         confidence
#                 })

#             except Exception as e:

#                 output_queue.put({
#                     "class_name": "No Audio",
#                     "confidence": 0.0
#                 })

#                 print(
#                     f"[Audio] Error: {e}"
#                 )

#     except Exception as e:

#         print(
#             f"[Audio] Worker startup error: {e}"
#         )

#         output_queue.put({
#             "class_name": "No Audio",
#             "confidence": 0.0
#         })


# class AudioController:

#     def __init__(self, config):

#         audio_config = config.get(
#             "audio",
#             {}
#         )

#         self.sample_rate = int(
#             audio_config.get(
#                 "sample_rate",
#                 16000
#             )
#         )

#         self.chunk_duration = float(
#             audio_config.get(
#                 "chunk_duration_sec",
#                 0.975
#             )
#         )

#         self.input_queue = mp.Queue()
#         self.output_queue = mp.Queue()

#         self.process = None

#         self.latest = {
#             "class_name": "No Audio",
#             "confidence": 0.0
#         }

#     def start(self):

#         print(
#             "[Audio] Starting process..."
#         )

#         self.process = mp.Process(
#             target=yamnet_worker,
#             args=(
#                 self.input_queue,
#                 self.output_queue,
#                 self.sample_rate,
#                 self.chunk_duration
#             ),
#             daemon=True
#         )

#         self.process.start()

#     def get_latest(self):

#         try:

#             while True:

#                 self.latest = (
#                     self.output_queue
#                     .get_nowait()
#                 )

#         except queue.Empty:
#             pass

#         except Exception:
#             pass

#         return self.latest

#     def stop(self):

#         if (
#             self.process is not None
#             and self.process.is_alive()
#         ):

#             self.process.terminate()

#             self.process.join(
#                 timeout=2
#             )








import queue
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np


def yamnet_worker(
    input_queue,
    output_queue,
    sample_rate,
    chunk_duration,
    device_index,
    classifier_path,
    class_map_path
):
    try:
        import sounddevice as sd
        import tensorflow as tf
        import tensorflow_hub as hub

        print("[Audio] Loading YAMNet...")
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

        print(f"[Audio] Loading classifier: {classifier_path}")
        classifier_file = Path(classifier_path)

        if not classifier_file.is_absolute():
            classifier_file = Path.cwd() / classifier_file

        if not classifier_file.is_file():
            raise FileNotFoundError(
                f"Audio classifier not found: {classifier_file}"
            )

        with open(classifier_file, "rb") as f:
            classifier = pickle.load(f)

        print(f"[Audio] Classifier type: {type(classifier).__name__}")

        # อ่านชื่อคลาสจาก classifier ที่ผ่านการฝึกแล้ว
        if hasattr(classifier, "classes_"):
            class_names = [
                str(name) for name in classifier.classes_
            ]
        else:
            class_names = None

        print(f"[Audio] Classes: {class_names}")
        print("[Audio] Models ready")

        block_size = int(sample_rate * chunk_duration)

        while True:
            try:
                audio = sd.rec(
                    block_size,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    device=device_index
                )
                sd.wait()
                audio = audio.reshape(-1)

                if len(audio) == 0:
                    output_queue.put({
                        "class_name": "No Audio",
                        "confidence": 0.0
                    })
                    continue

                # 1. สกัด YAMNet embeddings
                _, embeddings, _ = yamnet(
                    tf.convert_to_tensor(audio, dtype=tf.float32)
                )
                embeddings_np = embeddings.numpy()

                if embeddings_np.size == 0:
                    output_queue.put({
                        "class_name": "No Audio",
                        "confidence": 0.0
                    })
                    continue

                # 2. จำแนกเสียงด้วยโมเดลจาก configAI.json
                if hasattr(classifier, "predict_proba"):
                    probabilities = classifier.predict_proba(
                        embeddings_np
                    )
                    mean_probabilities = np.mean(
                        probabilities, axis=0
                    )
                    class_id = int(np.argmax(mean_probabilities))
                    confidence = float(mean_probabilities[class_id])

                    if class_names is not None:
                        class_name = class_names[class_id]
                    else:
                        class_name = str(class_id)

                else:
                    predictions = classifier.predict(embeddings_np)

                    # เลือกคลาสที่ทำนายบ่อยที่สุดใน chunk
                    labels, counts = np.unique(
                        predictions, return_counts=True
                    )
                    index = int(np.argmax(counts))
                    class_name = str(labels[index])
                    confidence = float(
                        counts[index] / len(predictions)
                    )

                output_queue.put({
                    "class_name": class_name,
                    "confidence": confidence
                })

            except Exception as e:
                print(f"[Audio] Error: {e}")
                output_queue.put({
                    "class_name": "No Audio",
                    "confidence": 0.0
                })

    except Exception as e:
        print(f"[Audio] Worker startup error: {e}")
        output_queue.put({
            "class_name": "No Audio",
            "confidence": 0.0
        })


class AudioController:
    def __init__(self, config):
        audio_config = config.get("audio", {})
        model_config = config.get("models", {}).get(
            "audio_classifier", {}
        )

        self.sample_rate = int(
            audio_config.get("sample_rate", 16000)
        )
        self.chunk_duration = float(
            audio_config.get("chunk_duration_sec", 0.975)
        )
        self.device_index = audio_config.get("device_index")

        self.classifier_path = model_config.get("path")
        self.class_map_path = model_config.get("class_map_path")

        if not self.classifier_path:
            raise ValueError(
                "Missing models.audio_classifier.path in configAI.json"
            )

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.process = None

        self.latest = {
            "class_name": "No Audio",
            "confidence": 0.0
        }

    def start(self):
        print("[Audio] Starting process...")

        self.process = mp.Process(
            target=yamnet_worker,
            args=(
                self.input_queue,
                self.output_queue,
                self.sample_rate,
                self.chunk_duration,
                self.device_index,
                self.classifier_path,
                self.class_map_path
            ),
            daemon=True
        )
        self.process.start()

    def get_latest(self):
        try:
            while True:
                self.latest = self.output_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass

        return self.latest

    def stop(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)