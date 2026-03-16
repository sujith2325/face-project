from deepface import DeepFace
import numpy as np

# image to test
img_path = r"C:\face_project\img1.jpeg"

# get embedding of test image
emb_test = DeepFace.represent(
    img_path,
    model_name="ArcFace",
    detector_backend="retinaface"
)[0]["embedding"]

# get embedding of dataset image
emb_dataset = DeepFace.represent(
    r"C:\face_project\dataset\Sujith\img1.jpeg",
    model_name="ArcFace",
    detector_backend="retinaface"
)[0]["embedding"]

distance = np.linalg.norm(np.array(emb_test) - np.array(emb_dataset))
print("Distance:", distance)

if distance < 0.8:
    print("✅ SAME PERSON")
else:
    print("❌ DIFFERENT PERSON")

print("Distance:", distance)