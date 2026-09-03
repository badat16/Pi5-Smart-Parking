"""
Driver Face Recognition Engine for Raspberry Pi 5.
Powered by InsightFace buffalo_sc & multi-threaded ONNX Runtime CPU.
"""

from typing import List, Optional, Tuple, Any
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(
        self,
        model_name: str = "buffalo_sc",
        det_thresh: float = 0.50,
        det_size: Tuple[int, int] = (320, 320),
        min_face_size: int = 45,
        num_threads: int = 4,
    ):
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.min_face_size = min_face_size
        self.num_threads = num_threads

        print(f"[FaceEngine] Initializing InsightFace '{model_name}' (CPU, threads={num_threads})...")

        # Configure ONNX Runtime thread count for Raspberry Pi 5
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.app = FaceAnalysis(
            name=self.model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(
            ctx_id=0,
            det_thresh=self.det_thresh,
            det_size=self.det_size,
        )
        print("[FaceEngine] InsightFace engine ready.")

    def analyze(self, frame: np.ndarray) -> List[Any]:
        """Detect and extract embeddings for all faces in frame."""
        if frame is None or frame.size == 0:
            return []
        return self.app.get(frame)

    def choose_primary_face(self, faces: List[Any]) -> Optional[Any]:
        """
        Select the single most prominent, clear face in the frame.
        Rejects if multiple people or face is too small / blurry.
        """
        if len(faces) == 0:
            return None

        # Filter faces by minimum bounding box dimensions
        valid_faces = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            w = x2 - x1
            h = y2 - y1
            if min(w, h) >= self.min_face_size:
                valid_faces.append(face)

        if len(valid_faces) == 1:
            return valid_faces[0]

        # If exactly 1 face overall
        if len(faces) == 1:
            x1, y1, x2, y2 = faces[0].bbox.astype(int)
            if min(x2 - x1, y2 - y1) >= self.min_face_size:
                return faces[0]

        return None

    @staticmethod
    def get_embedding(face: Any) -> np.ndarray:
        """Extract and L2-normalize face feature embedding (512-d)."""
        if face is None or not hasattr(face, "normed_embedding"):
            raise ValueError("Invalid face object for embedding extraction.")
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Zero-norm face embedding.")
        return emb / norm

    @staticmethod
    def draw_faces(frame: np.ndarray, faces: List[Any], labels: Optional[List[str]] = None) -> np.ndarray:
        """Draw bounding boxes and status labels on detected faces."""
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            score = float(face.det_score)

            label = labels[i] if (labels and i < len(labels)) else f"Driver ({score:.2f})"
            color = (0, 255, 0) if "MATCH" in label or "Driver" in label else (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame
