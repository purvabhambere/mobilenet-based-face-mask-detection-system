import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.keras")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = maskNet.predict(face, verbose=0)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            confidence_text = f"{max(mask, withoutMask) * 100:.2f}%"

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.putText(frame,
                        f"{label}: {confidence_text}",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

            cv2.rectangle(frame,
                          (startX, startY),
                          (endX, endY),
                          color,
                          2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()