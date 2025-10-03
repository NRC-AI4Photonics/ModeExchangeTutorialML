# Inverse Design of Photonic Integrated Devices: Neural Network Tutorial

This repository provides Python code accompanying the tutorial paper on inverse photonic design using neural networks, as described in the associated manuscript.

---

## Overview

This tutorial is structured into clear steps corresponding to the sections of the paper, covering:

- **Forward Model**: Training a neural network model to predict TE coupling coefficients from geometry parameters.
  - Simple forward model (`N_MODELS = 1`, `AUGMENT_DATA = False`)
  - Simple forward model with data augmentation (`N_MODELS = 1`, `AUGMENT_DATA = True`)
  - Ensemble forward model (`N_MODELS > 1`, `AUGMENT_DATA = True` recommended)

- **Inverse Model**: Predicting geometry parameters from specified TE coupling coefficients.
  - Simple inverse neural network (without tandem)
  - Tandem inverse neural network (with pre-trained forward network)

The provided Python script allows you to reproduce these examples directly, using simple flags and parameters. Additionally, the folder `./Simulations_setup` contains example code and needed scripts to run Ansys Lumerical simulations for data generation
---

## Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/maldig/inverse-photonics-design.git
cd inverse-photonics-design
```

### Step 2: Environment Setup

Create and activate a Python virtual environment (recommended):

```bash
python -m venv env
source env/bin/activate   # Linux/Mac
.\env\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Files

Ensure the following CSV files are present in the root folder:

- **Training data**:
  - `data_v19.csv` (original)
  - `data_v19_aug.csv` (augmented)

- **Test data**:
  - `frontier_v19.csv`

- **Generation data (optional)**:
  - `filtered_points_3.csv`

---

## Running the Code: Neural Network Tutorial Sequence

Follow these instructions sequentially to reproduce results from the manuscript.

### Section 1: Training the Forward Model

**1a. Simple Forward Model**

Edit flags in `inverse_design.py`:

```python
TRAIN_FORWARD = True
N_MODELS = 1            # Simple model
AUGMENT_DATA = False    # Original data only
```

Then run:

```bash
python inverse_design.py
```

**1b. Simple Forward Model with Data Augmentation**

Edit flags:

```python
TRAIN_FORWARD = True
N_MODELS = 1
AUGMENT_DATA = True     # Augmented dataset
```

Then run:

```bash
python inverse_design.py
```

**1c. Ensemble Forward Model**

Edit flags:

```python
TRAIN_FORWARD = True
N_MODELS = 5            # or any number >1
AUGMENT_DATA = True     # Augmented dataset recommended
```

Then run again:

```bash
python inverse_design.py
```

---

### Section 2: Training the Inverse Model

**2a. Simple Inverse Neural Network (illustrative)**

Swap X and y manually in `inverse_design.py`:

```python
X_train_scaled, y_train_scaled = y_train_scaled, X_train_scaled
X_val_scaled, y_val_scaled = y_val_scaled, X_val_scaled
X_test_scaled, y_test_scaled = y_test_scaled, X_test_scaled
```

Edit flags:

```python
TRAIN_FORWARD = False
TRAIN_INVERSE = True
N_MODELS = 1
```

Run:

```bash
python inverse_design.py
```

**2b. Advanced Tandem Inverse Neural Network (Recommended)**

Ensure forward model is trained first, then edit flags:

```python
TRAIN_FORWARD = False
TRAIN_INVERSE = True
N_MODELS = 5
```

Run:

```bash
python inverse_design.py
```

---

### Section 3: Generating Designs with the Trained Inverse Model

Set flags in the script:

```python
TRAIN_FORWARD = False
TRAIN_INVERSE = False
TEST_MODE = False
```

Place TE targets in `filtered_points_3.csv`. Run:

```bash
python inverse_design.py
```

Predicted design parameters saved:

```
results/generated_values_tandem_v19.csv
```

---

## Directory Structure after Execution

```
inverse-photonics-design/
├── data/                         # Optional
├── data_v19.csv
├── data_v19_aug.csv
├── frontier_v19.csv
├── filtered_points_3.csv
├── models/
│   ├── model_1.pt
│   ├── ...
│   ├── pytorch_ensemble_model_v19.pt
│   └── inverse_model_v19.pt
├── results/
│   └── generated_values_tandem_v19.csv
├── inverse_design.py
├── requirements.txt
└── README.md
```

---

## Customization

- To modify neural network architectures, edit `ForwardNet` and `InverseNet` classes.
- Experiment with training parameters (batch size, epochs, learning rate) at the top of `inverse_design.py`.

---

## Reference


---

