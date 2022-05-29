import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Tool import EyeTool

matplotlib.use("GTK4AGG")


class ResizeEyeTool(EyeTool):
    def __init__(self, img_path, faceDetector, faceMeshDetector):
        self.path = img_path
        self.faceDetector = faceDetector
        self.faceMeshDetector = faceMeshDetector
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.power = 1.25
        self.radius = (
            30 if self.image.shape[0] < 1000 or self.image.shape[1] < 1000 else 120
        )
        self.orig = self.image.copy()

    def apply(self, size=1.25, *args, **kwargs):
        """size: New Size Of Eyes
        \nkwargs:
            \nFile: Path For The Image To Be Modifed.
            \nRadius: The Region Around The Eye Where All Processing Are Done.
        """
        if "File" in kwargs:
            self.path = kwargs["File"]
            self.image = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
            self.orig = self.image.copy()
        if "Radius" in kwargs:
            self.radius = kwargs["Radius"]
        self.power = size

