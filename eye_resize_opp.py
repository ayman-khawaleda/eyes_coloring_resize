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

        results = self.faceDetector.process(self.image)
        rows, cols, _ = self.image.shape
        if not results.detections:
            raise Exception(f'No Faces Detected In Image With Path: "{self.path}".')

        for detection in results.detections:
            rbb = detection.location_data.relative_bounding_box
            rect_start_point = self.normaliz_pixel(rbb.xmin, rbb.ymin, cols, rows)
            rect_end_point = self.normaliz_pixel(
                rbb.xmin + rbb.width, rbb.ymin + rbb.height, cols, rows
            )
            faceROI = self.image[
                rect_start_point[1] : rect_end_point[1],
                rect_start_point[0] : rect_end_point[0],
                :,
            ].copy()
            mesh_result = self.faceMeshDetector.process(faceROI)
            h, w, _ = faceROI.shape
            eye_right,eye_left = self.__get_eyes_key_points(mesh_result,w,h)

    def __get_eyes_key_points(self, mesh, w, h):
        mp_face_mesh = mp.solutions.face_mesh
        right_eye_list, left_eye_list = [], []
        for face_landmarks in mesh.multi_face_landmarks:
            for tup in mp_face_mesh.FACEMESH_RIGHT_EYE:
                sor_idx, tar_idx = tup
                source = face_landmarks.landmark[sor_idx]
                target = face_landmarks.landmark[tar_idx]
                rel_source = (int(source.x * w), int(source.y * h))
                rel_target = (int(target.x * w), int(target.y * h	))
                right_eye_list.append(rel_source)
                right_eye_list.append(rel_target)

            for tup in mp_face_mesh.FACEMESH_LEFT_EYE:
                sor_idx, tar_idx = tup
                source = face_landmarks.landmark[sor_idx]
                target = face_landmarks.landmark[tar_idx]
                rel_source = (int(source.x * w), int(source.y * h))
                rel_target = (int(target.x * w), int(target.y * h))
                left_eye_list.append(rel_source)
                left_eye_list.append(rel_target)
            right_eye_list.sort(key=lambda x: x[1])
            right_eye_minh, right_eye_maxh = right_eye_list[0], right_eye_list[-1]
            right_eye_list.sort(key=lambda x: x[0])
            right_eye_minw, right_eye_maxw = right_eye_list[0], right_eye_list[-1]
            right_eye = (
                (right_eye_minw[0] + right_eye_maxw[0]) // 2,
                (right_eye_minh[1] + right_eye_maxh[1]) // 2,
            )

            left_eye_list.sort(key=lambda x: x[1])
            left_eye_minh, left_eye_maxh = left_eye_list[0], left_eye_list[-1]
            left_eye_list.sort(key=lambda x: x[0])
            left_eye_minw, left_eye_maxw = left_eye_list[0], left_eye_list[-1]
            left_eye = (
                (left_eye_minw[0] + left_eye_maxw[0]) // 2,
                (left_eye_minh[1] + left_eye_maxh[1]) // 2,
            )
            return right_eye, left_eye
