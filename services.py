from __future__ import annotations

import os
import threading
from typing import Optional

import faiss

import utils


class RecognitionService:
    _instance: Optional["RecognitionService"] = None

    def __init__(self) -> None:
        self.index: faiss.Index = utils.load_or_create_index()

    @classmethod
    def instance(cls) -> "RecognitionService":
        if cls._instance is None:
            if cls._instance is None:
                cls._instance = RecognitionService()
        return cls._instance

    def reload_index(self) -> int:
        self.index = utils.load_or_create_index()
        return self.index.ntotal

    def recognize_image_path(self, image_path: str) -> list[dict]:
        # Detect faces first, then classify/match
        faces = utils.detect_faces_and_crop(image_path)
        results: list[dict] = []
        for face_img_bgr in faces:
            # Compute embedding via tmp write path (DeepFace needs path)
            tmp_path = utils.save_numpy_image_temp(face_img_bgr)
            emb = utils.compute_embedding_from_path(tmp_path)
            try:
                if self.index.ntotal == 0:
                    similarity = 0.0
                    match_idx = -1
                else:
                    D, I = self.index.search(emb, k=1)
                    similarity = float(D[0][0])
                    match_idx = int(I[0][0])

                is_match = similarity > utils.THRESHOLD
                if not is_match:
                    # add to DB
                    new_path = utils.save_new_customer_face(face_img_bgr)
                    self.index.add(emb)
                    faiss.write_index(self.index, utils.FAISS_INDEX_PATH)
                    results.append(
                        {
                            "is_returning": False,
                            "similarity": similarity,
                            "index": match_idx,
                            "saved_path": new_path,
                        }
                    )
                else:
                    results.append(
                        {
                            "is_returning": True,
                            "similarity": similarity,
                            "index": match_idx,
                        }
                    )
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return results


