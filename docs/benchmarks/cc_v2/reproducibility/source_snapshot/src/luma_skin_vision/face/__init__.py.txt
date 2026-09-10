"""Optional user-supplied YuNet ONNX adapter. No weights downloaded implicitly."""

from pathlib import Path

import numpy as np

from luma_skin_vision.data import sha256


class YuNet:
    def __init__(self, model_path, expected_sha256):
        import cv2

        if sha256(model_path) != expected_sha256:
            raise ValueError("detector checksum mismatch")
        self.detector = cv2.FaceDetectorYN.create(str(Path(model_path)), "", (320, 320), 0.9)

    def detect(self, rgb):
        height, width = rgb.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(np.uint8(np.clip(rgb[..., ::-1], 0, 1) * 255))
        if faces is None or len(faces) != 1:
            raise ValueError("exactly one face required")
        face = faces[0]
        return {
            "bbox": face[:4].tolist(),
            "landmarks": face[4:14].reshape(5, 2).tolist(),
            "detector_score": float(face[14]),
        }
