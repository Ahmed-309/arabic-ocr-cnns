# %% [markdown]
# # Simple CNN for receipt regions
# This model classifies an **already cropped** region as `shop`, `item`,
# `date_time`, or `total`. It does not find regions or read their text.
# The XML provides the crop coordinates and labels; `receipts.csv` only lists
# images. The `boxes` folder is not used because it contains annotated previews.
# There are only 20 receipts, so treat this as a learning baseline.
# Run all cells using the project's `.venv` Python environment.

# %%
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
DATA_DIR = Path(r"D:\python\CNNs\archive")
CLASS_NAMES = ["shop", "item", "date_time", "total"]
IMAGE_SIZE = (48, 192)  # height, width

# %% [markdown]
# ## Load the labeled crops
# Convert each crop to grayscale and pad it to a fixed size while keeping its
# proportions. Box coordinates are axis-aligned; small XML rotation values are
# ignored in this baseline. The receipt polygons are not classification labels.

# %%
def prepare_crop(crop):
    crop = ImageOps.pad(
        crop.convert("L"), IMAGE_SIZE[::-1],
        method=Image.Resampling.BILINEAR, color=255,
    )
    return np.asarray(crop, dtype=np.float32)[..., np.newaxis] / 255.0


images, labels, receipt_ids = [], [], []
root = ET.parse(DATA_DIR / "annotations.xml").getroot()
for receipt in root.findall("image"):
    with Image.open(DATA_DIR / receipt.attrib["name"]) as photo:
        for box in receipt.findall("box"):
            label = box.attrib["label"]
            if label not in CLASS_NAMES:
                continue
            bounds = tuple(float(box.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr"))
            images.append(prepare_crop(photo.crop(bounds)))
            labels.append(CLASS_NAMES.index(label))
            receipt_ids.append(receipt.attrib["id"])

X = np.stack(images)
y = np.asarray(labels)
receipt_ids = np.asarray(receipt_ids)
print(f"{len(X)} crops from {len(np.unique(receipt_ids))} receipts")
print(dict(zip(CLASS_NAMES, np.bincount(y, minlength=len(CLASS_NAMES)).tolist())))

# %% [markdown]
# ## Split by receipt and inspect examples
# Hold out 20% of receipts. Splitting individual crops would let regions from the
# same photograph leak into both sets. The held-out set is used for validation
# and early stopping, so its scores are not an independent final test result.

# %%
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y, groups=receipt_ids))
assert set(receipt_ids[train_idx]).isdisjoint(receipt_ids[val_idx])
assert set(y[train_idx]) == set(range(len(CLASS_NAMES)))
assert set(y[val_idx]) == set(range(len(CLASS_NAMES)))
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
print(f"Training: {len(X_train)} crops; validation: {len(X_val)} crops")
print("Training receipt IDs:", np.unique(receipt_ids[train_idx]))
print("Validation receipt IDs:", np.unique(receipt_ids[val_idx]))

fig, axes = plt.subplots(2, 4, figsize=(12, 4))
for col, name in enumerate(CLASS_NAMES):
    for row, index in enumerate(np.flatnonzero(y == col)[:2]):
        axes[row, col].imshow(X[index, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[row, col].set_title(name)
        axes[row, col].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Build and train a small CNN
# Two convolution layers learn visual patterns; the final softmax returns four
# class probabilities. Class weights compensate for the many more `item` crops.
# Early stopping restores the weights with the lowest validation loss.

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMAGE_SIZE, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16,
    class_weight=dict(enumerate(weights)),
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True,
    )],
    verbose=2,
)

# %% [markdown]
# ## Evaluate all four classes
# Compare accuracy to an always-`item` baseline. The per-class report and
# confusion matrix reveal whether the model learns the less common regions.

# %%
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
predicted = model.predict(X_val, verbose=0).argmax(axis=1)
majority_class = np.bincount(y_train).argmax()
print(f"Validation accuracy: {val_accuracy:.1%}")
print(f"Always-{CLASS_NAMES[majority_class]} baseline: {np.mean(y_val == majority_class):.1%}")
print(classification_report(
    y_val, predicted, labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES, zero_division=0,
))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["loss"], label="Training")
axes[0].plot(history.history["val_loss"], label="Validation")
axes[0].set(xlabel="Epoch (starting at 0)", ylabel="Loss")
axes[0].legend()
ConfusionMatrixDisplay.from_predictions(
    y_val, predicted, labels=np.arange(len(CLASS_NAMES)),
    display_labels=CLASS_NAMES, ax=axes[1], colorbar=False,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save and try a crop
# `predict_region` expects a PIL image cropped around one region. For a new
# receipt, supply the region coordinates manually. Automatic localization and
# text reading would require additional models and more training data.

# %%
OUTPUT_DIR = DATA_DIR.parent
model.save(OUTPUT_DIR / "receipt_region_cnn.keras")
(OUTPUT_DIR / "receipt_region_classes.json").write_text(
    json.dumps(CLASS_NAMES, indent=2), encoding="utf-8",
)


def predict_region(crop):
    probabilities = model.predict(prepare_crop(crop)[np.newaxis], verbose=0)[0]
    best = int(probabilities.argmax())
    return CLASS_NAMES[best], float(probabilities[best])


example = int(val_idx[0])
probabilities = model.predict(X[example:example + 1], verbose=0)[0]
print("Actual:", CLASS_NAMES[y[example]])
print("Predicted:", CLASS_NAMES[int(probabilities.argmax())])
print("Saved model to:", OUTPUT_DIR / "receipt_region_cnn.keras")
