import os
import tempfile
from datetime import datetime

import cv2
import faiss
import numpy as np
from deepface import DeepFace


# Shared constants
MODEL_NAME = "Facenet512"
CUSTOMER_DIR = "known_faces"
FAISS_INDEX_PATH = "customer_index.faiss"
THRESHOLD = 0.7  # cosine similarity threshold for "same person"




def ensure_customer_dir_exists() -> None:
    if not os.path.isdir(CUSTOMER_DIR):
        os.makedirs(CUSTOMER_DIR, exist_ok=True)

def load_or_create_index() -> faiss.Index:
    ensure_customer_dir_exists()
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'. Expected an existing vector DB."
        )
    print(" Loading existing FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index


def save_numpy_image_temp(image_bgr: np.ndarray) -> str:
    # Save to a temporary file path for DeepFace.represent which expects a path
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="_tmp_face_")
    os.close(fd)
    cv2.imwrite(tmp_path, image_bgr)
    return tmp_path


def compute_embedding_from_path(img_path: str) -> np.ndarray:
    rep = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)
    emb = np.array(rep[0]["embedding"], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def detect_faces_and_crop(
    img_path: str,
    edge_margin_ratio: float = 0.005,
    profile_threshold: float = 0.6,
) -> list[np.ndarray]:
    detections = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend="retinaface",
        enforce_detection=False,
    )
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not load image.")
    img_h, img_w = img.shape[:2]

    cropped_faces: list[np.ndarray] = []
    print(f" Total detections before filtering: {len(detections)}")
    for i, face in enumerate(detections):
        region = face.get("facial_area", {})
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        fw = int(region.get("w", 0))
        fh = int(region.get("h", 0))

        if fw <= 0 or fh <= 0:
            continue

        # Edge filter
        if (
            x < img_w * edge_margin_ratio
            or y < img_h * edge_margin_ratio
            or (x + fw) > img_w * (1 - edge_margin_ratio)
            or (y + fh) > img_h * (1 - edge_margin_ratio)
        ):
            print(f" Face {i} near image edge — skipped.")
            continue

        crop = img[y : y + fh, x : x + fw]
        if crop.size == 0:
            continue

        # Profile filter using landmarks (if available)
        landmarks = face.get("facial_area", {}) 
        if landmarks:
            left_eye = np.array(landmarks.get("left_eye", [0, 0]))
            right_eye = np.array(landmarks.get("right_eye", [0, 0]))
            nose = np.array(landmarks.get("nose", [0, 0]))

            eye_distance = float(np.linalg.norm(left_eye - right_eye))
            nose_to_center = abs((float(left_eye[0]) + float(right_eye[0])) / 2.0 - float(nose[0]))
            asymmetry_ratio = nose_to_center / eye_distance if eye_distance > 0 else 1.0
            if asymmetry_ratio > profile_threshold:
                print(
                    f" Face {i} likely side profile (asymmetry={asymmetry_ratio:.2f}) — skipped."
                )
                continue

        cropped_faces.append(crop)

    print(f" Total valid faces after filtering: {len(cropped_faces)}")
    return cropped_faces




