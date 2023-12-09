import cv2
import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
from pydub import AudioSegment

class ObjectDetectionModule:
    def __init__(self, weights_path, cfg_path, classes_path):
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(320, 320), scale=1/255)

        with open(classes_path) as file_object:
            self.classes = [class_name.strip() for class_name in file_object.readlines()]

    def detect_objects(self, frame):
        return self.model.detect(frame)

    def draw_objects(self, frame, class_ids, scores, bboxes):
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            x, y, w, h = bbox
            class_name = self.classes[class_id]
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

class AudioRecognitionModule:
    def __init__(self, model_path, labels_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.waveform_input_index = self.input_details[0]['index']
        self.output_details = self.interpreter.get_output_details()
        self.scores_output_index = self.output_details[0]['index']

        self.labels = [label.strip() for label in open(labels_path).readlines()]

    def classify_audio(self, audio):
        audio_float = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        y_resampled = librosa.resample(audio_float, orig_sr=44100, target_sr=16000)
        y_norm = librosa.util.normalize(y_resampled)

        self.interpreter.resize_tensor_input(self.waveform_input_index, [y_norm.size], strict=False)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.waveform_input_index, y_norm)
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(self.scores_output_index)
        top_class_index = scores.argmax()

        return self.labels[top_class_index]

    def capture_and_classify_audio(self, duration=15, sr=44100):
        print("Recording...")
        audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype=np.int16)
        sd.wait()
        print("Recording finished.")
        return audio.flatten()

    def save_as_mp3(self, audio, filename="recorded_audio.mp3", sr=44100):
        audio = AudioSegment(
            audio.tobytes(),
            frame_rate=sr,
            sample_width=audio.dtype.itemsize,
            channels=1
        )
        audio.export(filename, format="mp3")
        print(f"Audio saved as {filename}")

def main():
    object_detection_module = ObjectDetectionModule(
        r"C:\Users\USER\Downloads\PCL\Object Detection\dnn_model\yolov4-tiny.weights",
        r"C:\Users\USER\Downloads\PCL\Object Detection\dnn_model\yolov4-tiny.cfg",
        r"C:\Users\USER\Downloads\PCL\Object Detection\dnn_model\classes.txt"
    )
    audio_recognition_module = AudioRecognitionModule(
        r"C:\Users\USER\Downloads\PCL\Speech Rec\lite-model_yamnet_classification_tflite_1.tflite",
        r'C:\Users\USER\Downloads/pcl\Speech Rec\labels.txt'
    )

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from the video stream
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret or frame is None:
            print("Error capturing frame")
            break

        # Perform audio recognition
        captured_audio = audio_recognition_module.capture_and_classify_audio()
        classification_result = audio_recognition_module.classify_audio(captured_audio)

        # If the audio is classified as something specific, trigger object detection
        if classification_result == "desired_class":
            class_ids, scores, bboxes = object_detection_module.detect_objects(frame)
            object_detection_module.draw_objects(frame, class_ids, scores, bboxes)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.destroyAllWindows()  # Comment out or remove this line

if __name__ == "__main__":
    main()
