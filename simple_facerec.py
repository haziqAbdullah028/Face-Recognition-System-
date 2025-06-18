import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster processing
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load face encodings from images in the given directory.
        :param images_path: Folder path containing known face images.
        """
        image_files = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(image_files)} encoding images found.")

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image: {img_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract the person's name from the image filename
            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            # Get the face encoding
            encodings = face_recognition.face_encodings(rgb_img)
            if not encodings:
                print(f"Warning: No face found in image: {img_path}")
                continue

            img_encoding = encodings[0]
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

        print("Encoding images loaded successfully.")

    def detect_known_faces(self, frame):
        """
        Detect and recognize known faces in a given video frame.
        :param frame: Frame from video feed (BGR format).
        :return: List of face locations and corresponding names.
        """
        # Resize and convert to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale face locations back to original frame size
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names
