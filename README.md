{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww13320\viewh10200\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Inverse Design of Photonic Integrated Devices: Neural Network Tutorial\
\
This repository provides Python code accompanying the tutorial paper on inverse photonic design using neural networks, as described in the associated manuscript.\
\
---\
\
## Overview\
\
These instructions are structured into steps corresponding to the sections of the tutorial paper, covering:\
\
- **Forward Model**: Training a neural network model to predict TE coupling coefficients from geometry parameters.\
  - Simple forward model (`N_MODELS = 1`, `AUGMENT_DATA = False`)\
  - Simple forward model with data augmentation (`N_MODELS = 1`, `AUGMENT_DATA = True`)\
  - Ensemble forward model (`N_MODELS > 1`, `AUGMENT_DATA = True` recommended)\
\
- **Inverse Model**: Predicting geometry parameters from specified TE coupling coefficients.\
  - Simple inverse neural network (without tandem)\
  - Tandem inverse neural network (with pre-trained forward network)\
\
The provided Python script allows you to reproduce these examples directly, using simple flags and parameters.\
\
---\
\
## Getting Started\
\
### Step 1: Clone the Repository\
\
```bash\
git clone https://github.com/your-repo/inverse-photonics-design.git\
cd inverse-photonics-design\
```\
\
### Step 2: Environment Setup\
\
Create and activate a Python virtual environment (recommended):\
\
```bash\
python -m venv env\
source env/bin/activate   # Linux/Mac\
.\\env\\Scripts\\activate    # Windows\
```\
\
Install dependencies:\
\
```bash\
pip install -r requirements.txt\
```\
\
---\
\
## Data Files\
\
Ensure the following CSV files are present in the root folder:\
\
- **Training data**:\
  - `data_v19.csv` (original)\
  - `data_v19_aug.csv` (augmented)\
\
- **Test data**:\
  - `frontier_v19.csv`\
\
- **Generation data (optional)**:\
  - `filtered_points_3.csv`\
\
---\
\
## Running the Code: Neural Network Tutorial Sequence\
\
Follow these instructions sequentially to reproduce results from the manuscript.\
\
### Section 1: Training the Forward Model\
\
**1a. Simple Forward Model**\
\
Edit flags in `inverse_design.py`:\
\
```python\
TRAIN_FORWARD = True\
N_MODELS = 1            # Simple model\
AUGMENT_DATA = False    # Original data only\
```\
\
Then run:\
\
```bash\
python inverse_design.py\
```\
\
**1b. Simple Forward Model with Data Augmentation**\
\
Edit flags:\
\
```python\
TRAIN_FORWARD = True\
N_MODELS = 1\
AUGMENT_DATA = True     # Augmented dataset\
```\
\
Then run:\
\
```bash\
python inverse_design.py\
```\
\
**1c. Ensemble Forward Model**\
\
Edit flags:\
\
```python\
TRAIN_FORWARD = True\
N_MODELS = 5            # or any number >1\
AUGMENT_DATA = True     # Augmented dataset recommended\
```\
\
Then run again:\
\
```bash\
python inverse_design.py\
```\
\
---\
\
### Section 2: Training the Inverse Model\
\
**2a. Simple Inverse Neural Network (illustrative)**\
\
Swap X and y manually in `inverse_design.py`:\
\
```python\
X_train_scaled, y_train_scaled = y_train_scaled, X_train_scaled\
X_val_scaled, y_val_scaled = y_val_scaled, X_val_scaled\
X_test_scaled, y_test_scaled = y_test_scaled, X_test_scaled\
```\
\
Edit flags:\
\
```python\
TRAIN_FORWARD = False\
TRAIN_INVERSE = True\
N_MODELS = 1\
```\
\
Run:\
\
```bash\
python inverse_design.py\
```\
\
**2b. Advanced Tandem Inverse Neural Network (Recommended)**\
\
Ensure forward model is trained first, then edit flags:\
\
```python\
TRAIN_FORWARD = False\
TRAIN_INVERSE = True\
N_MODELS = 5\
```\
\
Run:\
\
```bash\
python inverse_design.py\
```\
\
---\
\
### Section 3: Generating Designs with the Trained Inverse Model\
\
Set flags in the script:\
\
```python\
TRAIN_FORWARD = False\
TRAIN_INVERSE = False\
TEST_MODE = False\
```\
\
Place TE targets in `filtered_points_3.csv`. Run:\
\
```bash\
python inverse_design.py\
```\
\
Predicted design parameters saved:\
\
```\
results/generated_values_tandem_v19.csv\
```\
\
---\
\

## Customization

- To modify neural network architectures, edit `ForwardNet` and `InverseNet` classes.
- Experiment with training parameters (batch size, epochs, learning rate) at the top of `inverse_design.py`.

## Directory Structure after Execution\
\
```\
inverse-photonics-design/\
\uc0\u9500 \u9472 \u9472  data/                         # Optional\
\uc0\u9500 \u9472 \u9472  data_v19.csv\
\uc0\u9500 \u9472 \u9472  data_v19_aug.csv\
\uc0\u9500 \u9472 \u9472  frontier_v19.csv\
\uc0\u9500 \u9472 \u9472  filtered_points_3.csv\
\uc0\u9500 \u9472 \u9472  models/\
\uc0\u9474    \u9500 \u9472 \u9472  model_1.pt\
\uc0\u9474    \u9500 \u9472 \u9472  ...\
\uc0\u9474    \u9500 \u9472 \u9472  pytorch_ensemble_model_v19.pt\
\uc0\u9474    \u9492 \u9472 \u9472  inverse_model_v19.pt\
\uc0\u9500 \u9472 \u9472  results/\
\uc0\u9474    \u9492 \u9472 \u9472  generated_values_tandem_v19.csv\
\uc0\u9500 \u9472 \u9472  inverse_design.py\
\uc0\u9500 \u9472 \u9472  requirements.txt\
\uc0\u9492 \u9472 \u9472  README.md\
```\
\
---\
\
\
}
