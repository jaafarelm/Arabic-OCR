# ✒️ Arabic Handwritten Character Recognizer

A ResNet-style CNN that recognises handwritten Arabic characters, trained on
**70,910 samples** across **46 letter classes**, reaching **92.1% accuracy** on a
held-out test set.

**[▶️ Try the live demo](https://arabic-ocr-gfhz9meqcgp7dkwqtbsbgc.streamlit.app)** —
draw a character and the model predicts it.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **0.9205** |
| Test loss | 0.2380 |
| Train / Val accuracy | 0.924 / 0.928 |
| Classes | 46 base letters + digits |
| Training samples | 56,727 (of 70,910 total) |

Evaluated once on a stratified held-out test set that was never used for
training or tuning. Train and validation accuracy stay close throughout, so the
model is not overfitting.

---

## The project

This started as a university dissertation built entirely in Jupyter notebooks.
This repository is a deliberate rebuild: the same problem re-approached as
production-minded software, with every decision traced to evidence rather than
inherited assumption.

The interesting part was not the architecture — it was the debugging. Accuracy
went **0.37 → 0.48 → 0.71 → 0.85 → 0.90 → 0.92**, and each jump came from
diagnosing a specific cause:

**1. Validation instability (0.37 → 0.85).** Training loss fell smoothly while
validation loss spiked to 31. That signature — clean train, chaotic validation —
pointed at BatchNormalization: its running statistics were updating too slowly
to be usable at validation time. Setting `momentum=0.9` (from the 0.99 default),
together with a lower learning rate, stabilised it.

**2. Ill-posed label space (→ 0.90).** The source dataset labels every letter in
up to four positional forms (`Ha_Start`, `Ha_Middle`, `Ha_End`, `Ha_Isolated`).
In an isolated 32×32 glyph those forms are visually near-identical, so the model
was being asked a question the data could not answer. Collapsing them into one
class per letter (115 → 46) removed the ambiguity and roughly tripled the
samples per class.

**3. A corrupt label mapping (→ 0.92).** The benchmark looked healthy while the
live demo kept naming letters wrongly. Rendering samples against their labels
showed why: the dataset's shipped `label_mapping.csv` did not match its own
images — one class id was even occupied by an ingested `.ipynb_checkpoints`
folder. The model had been learning correct image groupings under wrong names,
which is invisible to a loss function but fatal to a demo. Worse, once a second
correctly-labelled dataset was merged, the two sources gave *contradictory*
labels for identical shapes.

The fix was to stop trusting the mapping file and derive labels from the
original directory structure, where the folder name **is** the label and
mislabelling is impossible by construction. Convergence went from ~20 epochs to
reach 87% down to **7 epochs to pass 90%**.

The throughline: every one of these was found by *rendering the data and looking
at it*, not by reasoning about it in the abstract.

---

## Architecture

A ResNet-style CNN (~2.8M parameters), chosen because residual connections keep
gradients flowing through deeper stacks:

```
Input 32×32×1
  ↓ Conv 3×3 (64) → BatchNorm → ReLU
  ↓ 2× ResidualBlock(64)              32×32
  ↓ 2× ResidualBlock(128, stride 2)   16×16
  ↓ 2× ResidualBlock(256, stride 2)    8×8
  ↓ GlobalAveragePooling → Dropout(0.5)
  ↓ Linear(256 → 46)
```

Each residual block computes `out = input + F(input)` via a skip connection.
Global average pooling replaces flatten-plus-dense, cutting parameters and
reducing overfitting.

**Training:** Adam (lr 1e-4), cross-entropy, batch size 128, early stopping on
validation loss, best-checkpoint saving. Augmentation applies rotation, shift,
scale and shear — **but never flips**, since a mirrored Arabic glyph is a
different or invalid letter.

---

## Data

| Dataset | Samples | Notes |
|---|---|---|
| [HMBD v1](https://github.com/HossamBalaha/HMBD-v1) | 54,110 | 115 positional-form folders → 46 base classes |
| [AHCD](https://www.kaggle.com/datasets/mloey1/ahcd1) | 16,800 | 28 base letters |
| **Merged** | **70,910** | 46 classes, avg 1,233 samples/class |

Merging the two was the point of the class collapse: AHCD has no positional
forms, so the label systems only align once HMBD's four forms per letter are
merged into one. Two datasets means two writing populations, which makes the
model less tied to any single style.

AHCD needed two fixes before it could be merged, both found by rendering the
images:
- **Transposed** — stored row/column swapped, so every letter was rotated 90°
  until fixed. Unnoticed, this would have silently corrupted training.
- **Inverted** — white ink on black, where HMBD is black on white.

---

## Setup

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Data — download and place:
#    HMBD Dataset/ folders  -> data/Dataset/
#    AHCD CSVs              -> data/ahcd_images.csv, data/ahcd_labels.csv

# 3. Build the label-correct cache from the folder structure (one time, ~4 min)
python src/build_dataset.py

# 4. Verify the merge
python src/data.py        # expect: MERGED 70,910 samples, 46 classes

# 5. Train (~4 min on GPU)
python src/train.py

# 6. Run the demo
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker build -t arabic-ocr .
docker run -p 8501:8501 arabic-ocr   # then open http://localhost:8501
```

The image is **CPU-only (511 MB)** by design: inference on one 32×32 image takes
milliseconds on CPU, so a CUDA image would add several gigabytes for no benefit.

---

## Structure

```
src/
  build_dataset.py   # one-time: folders → label-correct cached arrays
  data.py            # load cache, merge AHCD, stratified 3-way split
  preprocess.py      # SHARED by training and inference (crop-to-ink, centre, resize)
  model.py           # ResNet-style CNN
  train.py           # training loop, augmentation, early stopping
  predict.py         # load model, predict, top-k with letter names
  label_map.py       # class id → letter name, read from the cache
app/
  streamlit_app.py   # drawing-canvas demo
```

`preprocess.py` is deliberately shared: if training and inference transform
pixels differently, a good model still fails on real input.

---

## Known limitations

Written down rather than left to be discovered.

**Letters differing only by dots are the dominant error mode.** At 32×32,
dal (د) / zal (ذ) / raa (ر) differ by a single dot that survives downscaling as
only a few faint pixels. Most of the remaining ~8% error concentrates in these
diacritic-distinguished pairs. Higher input resolution is the obvious next
experiment.

**No writer-independent split.** Neither dataset ships writer identifiers, so
samples from the same writer may appear in both train and test. Reported
accuracy may therefore be slightly optimistic. This is a property of the source
data, not a choice.

**Drawing only, no photo upload.** Photo upload was implemented and then
deliberately removed. On a canvas the background and ink are controlled, so
input closely matches training data. Phone photos bring paper texture and uneven
lighting, which after downscaling leave a washed-out grey blob the model cannot
read. Rather than ship a path that fails, the demo keeps the one that works.
Adaptive (Otsu) thresholding would be the fix.

**Positional forms are not distinguished.** A deliberate trade: they are not
separable from an isolated glyph, so the task was reframed to one that is
well-posed.

---

## Roadmap

- Higher input resolution (48×48 / 64×64) to preserve diacritic dots
- Learning-rate scheduling to smooth late-training validation spikes
- Adaptive thresholding to restore photo upload
- Confusion-matrix analysis to quantify the confusable clusters
- Longer term: line-level recognition (CNN → BiLSTM → CTC) for connected script

---

## Stack

Python · PyTorch · Streamlit · Docker · NumPy · pandas · scikit-learn · Pillow
