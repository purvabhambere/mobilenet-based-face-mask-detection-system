from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from imutils import paths
import numpy as np
import os
import cv2

INIT_LR = 1e-4
EPOCHS = 40
BS = 32

print("[INFO] loading images...")
imagePaths = list(paths.list_images("dataset"))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Skipping unreadable image: {imagePath}")
        continue

    # FIX: Convert BGR -> RGB before preprocess_input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# FIX: LabelBinarizer gives (n,1) for binary — need (n,2) for softmax
if labels.shape[1] == 1:
    labels = np.hstack([1 - labels, labels])

print(f"[INFO] Classes: {lb.classes_}, Label shape: {labels.shape}")

(trainX, testX, trainY, testY) = train_test_split(
    data, labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

print("[INFO] building model...")
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# FIX: Freeze ALL base layers first, then unfreeze last 20
for layer in baseModel.layers:
    layer.trainable = False
for layer in baseModel.layers[-20:]:
    layer.trainable = True

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

classWeights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(trainY, axis=1)),
    y=np.argmax(trainY, axis=1)
)
classWeights = dict(enumerate(classWeights))

print("[INFO] training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    class_weight=classWeights
)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# FIX: Save as .h5 — compatible with TF 2.13 on Streamlit Cloud
print("[INFO] saving mask_detector.h5 ...")
model.save("mask_detector.h5")
print("[INFO] Done! Push mask_detector.h5 to GitHub.")