import mediapipe as mp
from Iris_Coloring_oop import ColoringEyeTool as CET

IMAGE_FILES = [
    r"Resources/man1.jpg",
    r"Resources/man2.jpg",
    r"Resources/woman1.jpg",
    r"Resources/woman2.jpg",
]

if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    for i, path in enumerate(IMAGE_FILES):
        cet = CET(path, face_mesh)
        cet.apply(color=(120, 60), saturation=25)
        cet.show_results()
