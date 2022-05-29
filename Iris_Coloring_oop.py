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

    def __are_eyes_open(self, face_landmarks, dist: int = 15):
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

    def __extract_iris_mask(self, face_landmarks, k: tuple = (3, 3), iter: int = 1):
        h, w, _ = self.image.shape
        right_th,left_th = None,None

        # Extract The Left (Iris & Eye) mask With Bin_Inv and Otsu
        # Than Generate Ellipse Mask have the same shape of Iris mask
        # Do bitwise_and Between Previous masks To make sure I have the right shape of Iris
        # Do MORPH_DILATE to expand the mask and getrid of balckholes (NOTE: Have better results than MORPH_OPENING)
        # End up with bitwise_and between Ellipse Mask and Previous mask
        if self.__is_left_open:
            for tup in mp.solutions.face_mesh.FACEMESH_LEFT_IRIS:
                sor_idx, _ = tup
                source = face_landmarks.landmark[sor_idx]
                rel_source = self.normaliz_pixel(source.x, source.y, w, h)
                self.__li_list.append(rel_source)
            self.left_iris = self.image[
                self.__li_list[1][1] : self.__li_list[2][1],
                self.__li_list[0][0] : self.__li_list[3][0],
                :,
            ].copy()
            _, left_th = cv2.threshold(
                cv2.cvtColor(self.left_iris, cv2.COLOR_RGB2GRAY),
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            lellipse = 255 * cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (left_th.shape[1], left_th.shape[0])
            ).astype(np.uint8)
            left_th = cv2.bitwise_and(left_th, lellipse)
            left_th = cv2.morphologyEx(left_th, cv2.MORPH_DILATE, k, iterations=iter)
            left_th = cv2.bitwise_and(left_th, lellipse)

        # Extract The Right (Iris & Eye) mask With Bin_Inv and Otsu
        if self.__is_right_open:
            for tup in mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS:
                sor_idx, _ = tup
                source = face_landmarks.landmark[sor_idx]
                rel_source = self.normaliz_pixel(source.x, source.y, w, h)
                self.__ri_list.append(rel_source)

            self.right_iris = self.image[
                self.__ri_list[3][1] : self.__ri_list[1][1],
                self.__ri_list[2][0] : self.__ri_list[0][0],
                :,
            ].copy()

            _, right_th = cv2.threshold(
                cv2.cvtColor(self.right_iris, cv2.COLOR_RGB2GRAY),
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            rellipse = 255 * cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (right_th.shape[1], right_th.shape[0])
            ).astype(np.uint8)
            right_th = cv2.bitwise_and(right_th, rellipse)
            right_th = cv2.morphologyEx(right_th, cv2.MORPH_DILATE, k, iterations=iter)
            right_th = cv2.bitwise_and(right_th, rellipse)

        return right_th, left_th
