import decimal
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from Tool import EyeTool


class ColoringEyeTool(EyeTool):
    mp_face_mesh = mp.solutions.face_mesh

    def __init__(self, img_path, faceMeshDetector):
        self.path = img_path
        self.faceMeshDetector = faceMeshDetector
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.orig = self.image.copy()

    def apply(self, color, saturation, *args, **kwargs):
        """
        \ncolor: [int,Tuple, List] [eye|eyes]'s Color Value In HSV Space Color.
        \nsaturation: [int, Tuple, List] Strength of Color.
        \nkwargs:
            \nFile: Path For The Image To Be Modifed.
        """
        if "File" in kwargs:
            self.path = kwargs["File"]
            self.image = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        results = self.faceMeshDetector.process(self.image)

        if not results.multi_face_landmarks:
            raise Exception(f'No Faces Detected In Image With Path: "{self.path}".')

    def __is_eyes_open(self, face_landmarks, dist: int = 15):
        right_min_p, right_max_p, left_min_p, left_max_p = (
            decimal.MAX_EMAX,
            0,
            decimal.MAX_EMAX,
            0,
        )
        mp_face_mesh = mp.solutions.face_mesh
        h, w, _ = self.image.shape
        for tup1, tup2 in zip(
            mp_face_mesh.FACEMESH_RIGHT_EYE, mp_face_mesh.FACEMESH_LEFT_EYE
        ):
            # Finding Both minimum & maximum values of right eyelid
            sor_idx, _ = tup1
            source = face_landmarks.landmark[sor_idx]
            norm = self.normaliz_pixel(source.x, source.y, w, h)
            if norm[1] > right_max_p:
                right_max_p = norm[1]
            if norm[1] < right_min_p:
                right_min_p = norm[1]

            # Finding Both minimum & maximum values of left eyelid
            sor_idx, _ = tup2
            source = face_landmarks.landmark[sor_idx]
            norm = self.normaliz_pixel(source.x, source.y, w, h)
            if norm[1] > left_max_p:
                left_max_p = norm[1]
            if norm[1] < left_min_p:
                left_min_p = norm[1]

        # Calculating the Distance between the two values
        return (right_max_p - right_min_p > dist, left_max_p - left_min_p > dist)

