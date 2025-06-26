import os
import pickle
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree  # (if needed later)
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----------------------- FLAGS -----------------------
TRAIN_FORWARD = True    # If False, load the pretrained forward (ensemble) model.
TRAIN_INVERSE = True    # If False, load the pretrained inverse model.
TEST_MODE = True# If True, run on the test set; else run on a generation file.

# ----------------------- PARAMETERS -----------------------
version = 19
TARGETS = 3         # Number of target (TE output) columns.
AUGMENT_DATA = True # Use augmented data if True.
N_MODELS = 5        # Number of models in the ensemble.
EPOCHS = 200        # Maximum training epochs for forward model.
BATCH_SIZE = 128    # Batch size for training forward model.
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Inverse (tandem) model parameters
INV_EPOCHS = 200     
INV_BATCH_SIZE = 128
INV_LEARNING_RATE = 0.001

# File names for saving/loading models and scalers.
forward_model_filename = f"pytorch_ensemble_model_v{version}.pt"
inverse_model_filename = f"inverse_model_v{version}.pt"
scaler_X_filename = f"scaler_X_v{version}.pkl"
scaler_y_filename = f"scaler_y_v{version}.pkl"

# Data files
train_file = f"data_v{version}_aug.csv" if AUGMENT_DATA else f"data_v{version}.csv"
test_file = f"frontier_v{version}.csv"

# ----------------------- DEVICE SETUP -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------- DATA LOADING & SCALING -----------------------
print("Loading data...")
df_train_val = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Split into design parameters (X) and targets (y)
X_train_val = df_train_val.iloc[:, :-TARGETS].values
y_train_val = df_train_val.iloc[:, -TARGETS:].values
X_test = df_test.iloc[:, :-TARGETS].values
y_test = df_test.iloc[:, -TARGETS:].values

# Split training/validation (85% train, 15% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=42
)

# Scale using MinMaxScaler so that all values lie in [0,1]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Save scalers for future reference
joblib.dump(scaler_X, scaler_X_filename)
joblib.dump(scaler_y, scaler_y_filename)
print("Scalers saved.")

DIM = X_train_scaled.shape[1]

# ----------------------- PYTORCH MODELS -----------------------
# Forward Model: 10 layers with 100 hidden units and final sigmoid.
class ForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForwardNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, 100))
        layers.append(nn.ReLU())
        for _ in range(9):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(100, output_dim))
        layers.append(nn.Sigmoid())  # to output in [0,1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Inverse Model: maps TE outputs to design parameters.
class InverseNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InverseNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, x):
        return self.model(x)

# ----------------------- TRAINING HELPERS -----------------------
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, patience):
    best_val_loss = float('inf')
    best_model_state = None
    trigger = 0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model

# ----------------------- DATA PREPARATION -----------------------
def get_dataloader(X, y, batch_size, shuffle=True):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_X, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

train_loader = get_dataloader(X_train_scaled, y_train_scaled, BATCH_SIZE)
val_loader = get_dataloader(X_val_scaled, y_val_scaled, BATCH_SIZE, shuffle=False)
test_loader = get_dataloader(X_test_scaled, y_test_scaled, BATCH_SIZE, shuffle=False)

# ----------------------- FORWARD MODEL (ENSEMBLE) -----------------------
ensemble_models = []
if TRAIN_FORWARD:
    print("\nTRAINING FORWARD (ENSEMBLE) MODEL...")
    for i in range(N_MODELS):
        print(f"\nTraining forward model {i+1}/{N_MODELS}...")
        model = ForwardNet(DIM, TARGETS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        model = train_model(model, optimizer, criterion, train_loader, val_loader, EPOCHS, EARLY_STOPPING_PATIENCE)
        # Save each individual model
        torch.save(model.state_dict(), f"model_{i+1}.pt")
        ensemble_models.append(model)
    
    # Reload models to ensure proper loading
    reloaded_models = []
    for i in range(N_MODELS):
        model = ForwardNet(DIM, TARGETS).to(device)
        model.load_state_dict(torch.load(f"model_{i+1}.pt"))
        model.eval()
        reloaded_models.append(model)
    
    # Build ensemble: average predictions from each model.
    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super(EnsembleModel, self).__init__()
            self.models = models

        def forward(self, x):
            preds = [model(x) for model in self.models]
            # Stack along new dimension and average.
            avg_pred = torch.mean(torch.stack(preds, dim=0), dim=0)
            return avg_pred

    ensemble_model = EnsembleModel(reloaded_models).to(device)
    # Optionally save the ensemble as a whole.
    torch.save(ensemble_model.state_dict(), forward_model_filename)
    print(f"Ensemble forward model saved as {forward_model_filename}.")
else:
    print("\nLOADING PRETRAINED FORWARD MODEL...")
    # Assume models are stored as ensemble weights; you may need to recreate models if storing separately.
    # For simplicity, we rebuild the ensemble with one base model and load state.
    base_model = ForwardNet(DIM, TARGETS).to(device)
    base_model.load_state_dict(torch.load(f"model_1.pt"))
    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super(EnsembleModel, self).__init__()
            self.models = models

        def forward(self, x):
            preds = [model(x) for model in self.models]
            return torch.mean(torch.stack(preds, dim=0), dim=0)
    ensemble_model = EnsembleModel([base_model]).to(device)
    ensemble_model.eval()

# Evaluate forward model on test set (scaled metrics)
print("\nEvaluating Forward Model on Test Set (scaled)...")
ensemble_model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = ensemble_model(batch_X)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(batch_y.numpy())
y_pred_scaled = np.vstack(all_preds)
y_test_scaled_all = np.vstack(all_targets)

test_loss = np.mean((y_pred_scaled - y_test_scaled_all)**2)
test_mae = np.mean(np.abs(y_pred_scaled - y_test_scaled_all))
print("Test Loss (MSE):", test_loss, "Test MAE:", test_mae)

# Inverse-transform predictions for reporting (forward model)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_scaled_all)
mae_per_param = np.mean(np.abs(y_pred - y_test_unscaled), axis=0)
mse_per_param = np.mean((y_pred - y_test_unscaled)**2, axis=0)
results_df_forward = pd.DataFrame({
    'Parameter': [f'TE {i}' for i in range(TARGETS)],
    'MAE': mae_per_param,
    'MSE': mse_per_param
})
print("\nForward Model Test Results (Original Units):")
print(results_df_forward)

# ----------------------- INVERSE MODEL (TANDEM NETWORK) -----------------------
def build_tandem_model(forward_net, inverse_net):
    # Freeze forward model parameters.
    for param in forward_net.parameters():
        param.requires_grad = False

    # Tandem model: input TE, inverse_net -> design params -> forward_net -> TE prediction.
    class TandemModel(nn.Module):
        def __init__(self, forward_net, inverse_net):
            super(TandemModel, self).__init__()
            self.forward_net = forward_net
            self.inverse_net = inverse_net

        def forward(self, x):
            design = self.inverse_net(x)
            te_pred = self.forward_net(design)
            return te_pred

    return TandemModel(forward_net, inverse_net)

if TRAIN_INVERSE:
    print("\nTRAINING INVERSE MODEL (TANDEM NETWORK)...")
    inverse_net = InverseNet(TARGETS, DIM).to(device)
    # Use the ensemble forward model and freeze it.
    ensemble_model.eval()  # already frozen if needed
    tandem_model = build_tandem_model(ensemble_model, inverse_net).to(device)
    
    optimizer_inv = optim.Adam(inverse_net.parameters(), lr=INV_LEARNING_RATE)
    criterion = nn.MSELoss()

    # Use training TE targets as both input and label (we want forward(inverse(TE)) ~ TE)
    tensor_y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    tensor_y_val = torch.tensor(y_val_scaled, dtype=torch.float32)
    train_inv_loader = DataLoader(TensorDataset(tensor_y_train, tensor_y_train), batch_size=INV_BATCH_SIZE, shuffle=True)
    val_inv_loader = DataLoader(TensorDataset(tensor_y_val, tensor_y_val), batch_size=INV_BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(INV_EPOCHS):
        tandem_model.train()
        train_losses = []
        for batch_in, batch_target in train_inv_loader:
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            optimizer_inv.zero_grad()
            te_pred = tandem_model(batch_in)
            loss = criterion(te_pred, batch_target)
            loss.backward()
            optimizer_inv.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # Validation
        tandem_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_in, batch_target in val_inv_loader:
                batch_in, batch_target = batch_in.to(device), batch_target.to(device)
                te_pred = tandem_model(batch_in)
                loss = criterion(te_pred, batch_target)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f"[Inverse] Epoch {epoch+1}/{INV_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_inverse_state = inverse_net.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Inverse model early stopping triggered.")
                break

    inverse_net.load_state_dict(best_inverse_state)
    # Save inverse model
    torch.save(inverse_net.state_dict(), inverse_model_filename)
    print(f"Inverse model saved as {inverse_model_filename}.")
else:
    print("\nLOADING PRETRAINED INVERSE MODEL...")
    inverse_net = InverseNet(TARGETS, DIM).to(device)
    inverse_net.load_state_dict(torch.load(inverse_model_filename))
    inverse_net.eval()

# ----------------------- RUN INVERSE MODEL -----------------------
if TEST_MODE:
    print("\nRunning Inverse Model on Test Set...")
    # Predict design parameters from TE outputs (test set)
    inverse_net.eval()
    all_inv_preds = []
    with torch.no_grad():
        tensor_y_test = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)
        inv_pred_scaled = inverse_net(tensor_y_test)
        all_inv_preds.append(inv_pred_scaled.cpu().numpy())
    inv_pred_scaled = np.vstack(all_inv_preds)
    final_predictions_inv = scaler_X.inverse_transform(inv_pred_scaled)
    X_test_unscaled = scaler_X.inverse_transform(X_test_scaled)
    mse_per_feature_inv = mean_squared_error(X_test_unscaled, final_predictions_inv, multioutput='raw_values')
    mae_per_feature_inv = mean_absolute_error(X_test_unscaled, final_predictions_inv, multioutput='raw_values')
    results_df_inverse = pd.DataFrame({
        'Feature': [f'param_{i}' for i in range(DIM)],
        'MSE': mse_per_feature_inv,
        'MAE': mae_per_feature_inv
    })
    print("\nInverse Model Test Results (Design Parameters in Original Units):")
    print(results_df_inverse)
else:
    print("\nRunning Inverse Model in Generation Mode...")
    # Load generation input file, which should contain the target values (TE outputs)
    input_data = pd.read_csv('filtered_points_3.csv', header=0, delimiter=',')
    input_data = input_data.apply(pd.to_numeric, errors='coerce').dropna()
    # Get target values (first three columns) from the input data.
    te_targets = input_data.iloc[:, :TARGETS]
    te_gen_scaled = scaler_y.transform(input_data.values)
    te_gen_tensor = torch.tensor(te_gen_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        inv_pred_scaled = inverse_net(te_gen_tensor).cpu().numpy()
    final_predictions_inv = scaler_X.inverse_transform(inv_pred_scaled)
    # Create a DataFrame: first the target values, then the generated design parameters.
    design_param_df = pd.DataFrame(final_predictions_inv, columns=[f'param_{i}' for i in range(DIM)])
    gen_output = pd.concat([te_targets.reset_index(drop=True), design_param_df.reset_index(drop=True)], axis=1)
    gen_filename = f'generated_values_tandem_v{version}.csv'
    gen_output.to_csv(gen_filename, index=False)
    print(f"Generated design parameters saved as {gen_filename}.")

# ----------------------- SAVE FINAL MODELS (if trained) -----------------------
if TRAIN_FORWARD:
    torch.save(ensemble_model.state_dict(), forward_model_filename)
    print(f"Forward ensemble model saved as {forward_model_filename}.")
if TRAIN_INVERSE:
    torch.save(inverse_net.state_dict(), inverse_model_filename)
    print(f"Inverse model saved as {inverse_model_filename}.")

print("\nProcess complete.")

