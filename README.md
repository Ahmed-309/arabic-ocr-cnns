# Arabic OCR CNNs

Two small image-classification experiments built with TensorFlow/Keras: a Vision
Transformer that recognizes single Arabic characters, and a CNN that labels cropped
regions of a shopping receipt. Both are learning projects, not production OCR. They
classify one pre-cropped image at a time and do not locate or read text.

## Features

- **Arabic character classifier (`arabic_vit.ipynb`)**: a small Vision Transformer
  with 2 attention blocks, 8×8 patches and 64×64 grayscale input. It predicts one of
  38 labels: 29 letters (including `لا`) and 9 Arabic-Indic digits (`٥` is not in
  the dataset).
  - The data is split 70/15/15 into train, validation and test sets, stratified by class.
  - Images load in batches with `tf.data`, so the dataset is never held in RAM all at once.
  - Training images get small random rotations and translations. There are no flips,
    because flipping can change which character an image shows.
  - Training uses early stopping and reduces the learning rate when validation loss stalls.
  - The notebook prints a classification report and a confusion matrix, and shows a
    grid of misclassified test images.
  - `predict_character(path)` returns the top 3 predictions with their softmax scores.
- **Receipt region classifier (`cnn2.ipynb.py`)**: a 2-layer CNN that labels a
  cropped receipt region as `shop`, `item`, `date_time` or `total`.
  - Crops come from CVAT XML bounding boxes and are padded to 48×192 grayscale
    without distortion.
  - The train/validation split is grouped by receipt, so crops from one photo never
    appear in both sets.
  - Class weights offset how common the `item` label is, and accuracy is reported
    next to an always-`item` baseline.
  - `predict_region(crop)` classifies a PIL image crop.
- **`cnn2.ipynb`**: a guided practice notebook with step-by-step instructions and
  empty cells for rebuilding the receipt CNN yourself.

## Installation

```bash
git clone https://github.com/Ahmed-309/arabic-ocr-cnns.git
cd arabic-ocr-cnns
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt jupyter
```

## Data

The datasets are **not included** in this repository. Place them like this:

```
arabic letters/
  Dataset/
    ا/  ب/  ت/  ...  ٩/       # one folder per class, PNG images inside
archive/
  annotations.xml              # CVAT 1.1 export with shop/item/date_time/total boxes
  images/0.jpg ... 19.jpg      # receipt photos referenced by the XML
```

Folder names under `Dataset/` become the class labels. The original Arabic
dataset had 94,269 images, and the receipt set has 20 annotated photos.

**Paths are hard-coded.** Before running, update `DATA_DIR` in each notebook
(`D:\python\CNNs\...` in the originals) to match your machine.

## Usage

Open `arabic_vit.ipynb` in Jupyter and run the cells in order. Training on the full
dataset can take a while on a CPU. When it finishes, the notebook writes
`arabic_vit.keras` and `arabic_vit_classes.json` next to the notebook.

`cnn2.ipynb.py` uses `# %%` cell markers. Run it cell by cell in VS Code's
interactive window, or run it as a script:

```bash
python cnn2.ipynb.py
```

It saves `receipt_region_cnn.keras` and `receipt_region_classes.json`.

### Using the pre-trained models

Both trained models are included in the repo. To load the receipt model:

```python
import json, tensorflow as tf
model = tf.keras.models.load_model("receipt_region_cnn.keras")
classes = json.load(open("receipt_region_classes.json"))
```

The ViT uses a custom `PatchEmbedding` layer. In a fresh kernel, run the notebook's
import cell and the `PatchEmbedding` class definition before calling
`keras.models.load_model("arabic_vit.keras")`. Get the label order from
`arabic_vit_classes.json`, not by re-scanning the dataset folder.

## Project Structure

| File | Purpose |
|------|---------|
| `arabic_vit.ipynb` | Trains and evaluates the Arabic character Vision Transformer |
| `arabic_vit.keras` | Trained ViT model |
| `arabic_vit_classes.json` | The 38 class labels, in the model's output order |
| `cnn2.ipynb.py` | Trains and evaluates the receipt region CNN (complete solution) |
| `cnn2.ipynb` | Practice notebook for building the receipt CNN step by step |
| `receipt_region_cnn.keras` | Trained receipt region model |
| `receipt_region_classes.json` | Receipt class labels: `shop`, `item`, `date_time`, `total` |

## Limitations

- The receipt model is trained on only 20 receipts, so treat it as a baseline. Its
  validation set is also used for early stopping, so validation scores are optimistic.
- The character split is by image. If the same writer appears in both the training
  and test sets, the test accuracy may overstate how well the model handles new
  handwriting.

## Requirements

- Developed with Python 3.14 and TensorFlow 2.22. Saving and loading the `.keras`
  files requires Keras 3, which comes with TensorFlow 2.16 and later.
- numpy, matplotlib, scikit-learn, pillow
