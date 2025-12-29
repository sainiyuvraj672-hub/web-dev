import cv2
import numpy as np
from matplotlib import pyplot as plt

# ============================================================
# 🔥🔥🔥 1. PATH TO YOUR MODEL FILES (EDIT THIS PART ONLY) 🔥🔥🔥
# ============================================================

prototxt_path = r"c:\Users\user\OneDrive\Desktop\Models\deploy.prototxt.txt"
model_path    = r"c:\Users\user\OneDrive\Desktop\Models\res10_300x300_ssd_iter_140000.caffemodel"

# ------------- 2. Load the DNN Face Detector -------------
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


# ------------- 3. Add Your Photos Here -------------
image_paths = [
    r"c:\Users\user\OneDrive\Desktop\My photos\Snapchat-2138548560.jpg",
    # You can add more images like:
    r"c:\Users\user\OneDrive\Desktop\My photos\Snapchat-1865151656.jpg",
]


# ------------- 4. Face Detection Function -------------
def detect_faces_dnn(img_bgr, conf_threshold=0.6):

    (h, w) = img_bgr.shape[:2]

    # Create input blob for DNN
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    img_out = img_bgr.copy()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            boxes.append((x1, y1, x2, y2))

            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_out, f"{confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img_out, boxes


# ------------- 5. Run On All Images -------------
for img_path in image_paths:
    print(f"\nProcessing: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Could not load image. Check file path.")
        continue

    img_det, boxes = detect_faces_dnn(img)

    print(f"Number of faces detected: {len(boxes)}")

    img_rgb = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)
    plt.title(f"{len(boxes)} face(s) detected (DNN)")
    plt.axis("off")
    plt.show()
