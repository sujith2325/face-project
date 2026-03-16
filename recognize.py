import cv2

@app.route("/recognize", methods=["POST"])
def recognize():
    img = request.files["image"]

    os.makedirs("static", exist_ok=True)
    img_path = "static/test.jpg"
    img.save(img_path)

    # Get embedding
    result = DeepFace.represent(
        img_path,
        model_name="ArcFace",
        detector_backend="retinaface"
    )[0]

    test_embedding = result["embedding"]
    face = result["facial_area"]

    best_match = None
    min_distance = 999

    for name, db_embedding in database.items():
        distance = np.linalg.norm(
            np.array(test_embedding) - np.array(db_embedding)
        )
        print("Distance with", name, "=", distance)
        print("Database contains:", list(database.keys()))
        print("Database:",database.keys())

        if distance < min_distance:
            min_distance = distance
            best_match = name

    image = cv2.imread(img_path)

    if min_distance < THRESHOLD:
        label = best_match
        color = (0, 255, 0)
    else:
        label = "Unknown"
        color = (0, 0, 255)

    # Draw rectangle around face
    cv2.rectangle(
        image,
        (face["x"], face["y"]),
        (face["x"] + face["w"], face["y"] + face["h"]),
        color,
        3
    )

    # Put name text
    cv2.putText(
        image,
        label,
        (face["x"], face["y"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    output_path = "static/output.jpg"
    cv2.imwrite(output_path, image)

    return render_template("result.html", image=output_path)
    