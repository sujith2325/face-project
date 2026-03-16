from flask import Flask, request, render_template
from deepface import DeepFace
import numpy as np
import cv2
import os
import pickle

app = Flask(__name__)

DB_PATH = "database.pkl"
THRESHOLD = 0.68

# Load database
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        database = pickle.load(f)
else:
    database = {}


# HOME PAGE (REGISTER)
@app.route("/")
def home():
    return render_template("register.html")

# RECOGNIZE PAGE
@app.route("/recognize_page")
def recognize_page():
    return render_template("recognize.html")

# VIEW DATABASE PAGE
@app.route("/database")
def view_database():
    persons = list(database.keys())
    return render_template("database.html", persons=persons)

# DELETE PERSON FROM DATABASE
@app.route("/delete/<name>", methods=["POST"])
def delete_person(name):
    if name in database:
        del database[name]
        with open(DB_PATH, "wb") as f:
            pickle.dump(database, f)
    return render_template("database.html", persons=list(database.keys()), message=f"'{name}' has been removed from the database.")


# REGISTER PERSON
@app.route("/register", methods=["POST"])
def register():
    name = request.form["name"].strip()
    img = request.files["image"]

    if not name:
        return render_template("register.html", error="Please enter a valid name.")

    # Save uploaded image persistently
    person_dir = os.path.join("static", "dataset", name)
    os.makedirs(person_dir, exist_ok=True)

    # Count existing images for this person to avoid overwriting
    existing = len(os.listdir(person_dir))
    img_filename = f"{existing + 1}.jpg"
    img_path = os.path.join(person_dir, img_filename)
    img.save(img_path)

    try:
        representations = DeepFace.represent(
            img_path,
            model_name="ArcFace",
            detector_backend="retinaface"
        )
        if not representations:
            os.remove(img_path)
            return render_template("register.html", error="No face detected in the image. Please try again with a clearer photo.")

        embedding = representations[0]["embedding"]

        if name not in database:
            database[name] = []

        database[name].append(embedding)

        with open(DB_PATH, "wb") as f:
            pickle.dump(database, f)

        return render_template("register.html", message=f"✅ '{name}' registered successfully! You can now recognize them.")
    except Exception as e:
        print(f"Registration error: {e}")
        if os.path.exists(img_path):
            os.remove(img_path)
        return render_template("register.html", error=f"Error processing image: {str(e)}")


# RECOGNIZE PERSON (Single image, identify all faces from DB)
@app.route("/recognize", methods=["POST"])
def recognize():
    img_file = request.files.get("image")

    if not img_file:
        return render_template("recognize.html", error="Please upload an image.")

    os.makedirs("static", exist_ok=True)

    img_path = "static/recognize_input.jpg"
    img_file.save(img_path)

    try:
        results = DeepFace.represent(
            img_path,
            model_name="ArcFace",
            detector_backend="retinaface"
        )
    except Exception as e:
        results = []
        print(f"Face detection error: {e}")

    image = cv2.imread(img_path)
    identified_people = []

    if not results:
        # No faces detected - just show the image with a message
        output_path = "static/recognize_output.jpg"
        cv2.imwrite(output_path, image)
        return render_template(
            "result.html",
            output_image="recognize_output.jpg",
            identified_people=[],
            no_face=True
        )

    for result in results:
        test_embedding = result["embedding"]
        face = result["facial_area"]

        if not database:
            label = "Unknown (DB Empty)"
            color = (0, 165, 255)  # Orange
            min_distance = -1
        else:
            best_match = None
            min_distance = float("inf")

            for name, embeddings in database.items():
                for db_embedding in embeddings:
                    distance = float(np.linalg.norm(
                        np.array(test_embedding) - np.array(db_embedding)
                    ))
                    print(f"Distance with '{name}' = {distance:.4f}")
                    if distance < min_distance:
                        min_distance = distance
                        best_match = name

            if min_distance < THRESHOLD:
                label = best_match
                color = (0, 220, 0)  # Green
            else:
                label = "Unknown"
                color = (0, 0, 220)  # Red

        # Draw bounding box
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

        # Draw filled label background
        label_text = f"{label} ({min_distance:.2f})" if min_distance >= 0 else label
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x, y - text_h - 15), (x + text_w + 10, y), color, -1)
        cv2.putText(
            image,
            label_text,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        identified_people.append({
            "name": label,
            "distance": round(min_distance, 4) if min_distance >= 0 else None,
            "matched": min_distance < THRESHOLD if min_distance >= 0 else False
        })

    output_path = "static/recognize_output.jpg"
    cv2.imwrite(output_path, image)

    return render_template(
        "result.html",
        output_image="recognize_output.jpg",
        identified_people=identified_people,
        no_face=False
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)