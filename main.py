import mediapipe as mp

IMAGE_FILES = [
    r"Resources/man1.jpg",
    r"Resources/man2.jpg",
    r"Resources/woman1.jpg",
    r"Resources/woman2.jpg",
]

if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detecation = mp.solutions.face_detection

    face_detecation = mp_face_detecation.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
