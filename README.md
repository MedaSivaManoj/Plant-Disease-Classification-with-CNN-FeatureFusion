# Plant Disease Image Detection

Classifies leaf images into **38 plant disease / healthy classes** (PlantVillage-style dataset) and compares three approaches side by side:

| Approach | Description |
|----------|-------------|
| **Gabor** | Hand-crafted texture + color features (Gabor filters, LBP, color histogram) fed to a Random Forest |
| **CNN** | Convolutional neural network trained end-to-end on 128×128 RGB images |
| **Fusion** | CNN deep features concatenated with Gabor features, classified by a stacking ensemble (Random Forest, SVC, Logistic Regression) |

A Streamlit app (`demo.py`) lets you upload a leaf image and see all three predictions with confidence scores.

## Project structure

```
.
├── demo.py                 # Streamlit demo comparing the three models
├── gabor/                  # Part 1: classical ML on handcrafted features
│   ├── cs611-project-part1.ipynb
│   ├── random_forest_model.pkl, scaler.pkl
│   └── X_val/X_test/y_*.npy
├── cnn/                    # Part 2: CNN classifier
│   ├── cs611-project-part2.ipynb
│   ├── plant_disease_model_new.keras
│   ├── class_names.json, history.json
├── fusion/                 # Part 3: CNN + Gabor feature fusion
│   ├── cs611-project-part3.ipynb
│   ├── plant_disease_model.keras
│   ├── scaler_fused.pkl, label_encoder.pkl, stacking_fused.pkl*
│   └── classification_report.txt, test_classification_report.txt
└── test_dataset/test1/     # Sample test images, one folder per class
```

\* `stacking_fused.pkl` is excluded from the repository by `.gitignore` due to its size (see [Notes](#notes)).

## Feature pipeline

All images are resized to 128×128. The classical models use:

- **Gabor** – 4 frequencies (0.1–0.4) × 4 orientations (0, π/4, π/2, 3π/4), with mean, variance, std and max per filter → 64 features
- **LBP** – uniform local binary pattern (P=8, R=1) histogram → 10 features
- **Color** – 8×8×8 BGR histogram → 512 features

The Gabor model uses all 586 features. The fusion model concatenates the CNN's `feature_output` layer with the 64 Gabor features, standardizes them, and passes them to the stacking classifier.

## Results

| Model | Notes |
|-------|-------|
| CNN | Final epoch: ~66% train / ~85% validation accuracy (`cnn/history.json`) |
| Fusion | **0.89 accuracy** on 7,016 test images (macro F1 0.89); **0.90** on the 608-image evaluation in `fusion/classification_report.txt` |

Per-class precision/recall is in [fusion/test_classification_report.txt](fusion/test_classification_report.txt). Weakest classes are Tomato Septoria leaf spot, Target Spot and Late blight (F1 ≈ 0.75–0.78); strongest are Corn healthy and Common rust (≈ 0.98–0.99).

## Setup

Requires Python 3.9+.

```bash
pip install streamlit tensorflow numpy opencv-python pillow scikit-image scikit-learn joblib pandas
```

## Run the demo

From the project root (model paths are relative):

```bash
streamlit run demo.py
```

Then upload a `.jpg`/`.png` leaf image (try one from `test_dataset/test1/`), pick its true label, and click **Run Prediction**.

## Retraining

Open the notebooks in order (they were written for Colab/Kaggle, so adjust dataset paths):

1. `gabor/cs611-project-part1.ipynb` – feature extraction, Random Forest / SVM / KNN with grid search
2. `cnn/cs611-project-part2.ipynb` – CNN training
3. `fusion/cs611-project-part3.ipynb` – feature fusion and stacking ensemble

## Notes

- `.gitignore` excludes `*.pkl` and `*.npy`, but some of those files (scalers, label encoder, Random Forest) are required by `demo.py`. If you clone from Git and they are missing, regenerate them with the notebooks.
- `fusion/stacking_fused.pkl` and `gabor/X_train.npy` are not included; run the Part 3 notebook to recreate the stacking model before launching the demo.
- Input images must be preprocessed exactly as in training (resize to 128×128, BGR→RGB for the CNN); `demo.py` already does this.
