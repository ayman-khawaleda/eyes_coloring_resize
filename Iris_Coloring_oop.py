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


