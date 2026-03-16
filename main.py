from deepface import DeepFace

result = DeepFace.verify(
    r"C:\face_project\img1.jpeg",
    r"C:\face_project\img2.jpeg"
    model_name="ArcFace",
    detector_backend="retinaface"
)

print(result)