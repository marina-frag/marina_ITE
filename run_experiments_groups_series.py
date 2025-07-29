# cmd.exe /c run_experiments.bat
# chmod +x run_experiments.sh and ./run_experiments.sh

from collections import defaultdict


from concurrent.futures import ThreadPoolExecutor
from functools import partial


import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import shutil              
import argparse
import torch.optim as optim
import numpy as np
import torch.nn as nn
import seaborn as sns
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,        # ← add this
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

# 1) Increase global font size
plt.rcParams.update({
    'font.size': 20,           # base font size for text
    'axes.titlesize': 22,      # title size
    'axes.labelsize': 20,      # x/y label size
    'xtick.labelsize': 18,     # x tick labels
    'ytick.labelsize': 18,     # y tick labels
    'legend.fontsize': 20,     # legend text
    'figure.titlesize': 24,    # figure title if you set one
})

def my_cuda():
# Setting up the device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    devNumber = torch.cuda.current_device() if torch.cuda.is_available() else None
    devName = torch.cuda.get_device_name(devNumber) if devNumber is not None else None
    if devNumber is not None:
        print(f"Current CUDA device number: {devNumber}")
        print(f"Device name: {devName}")   
    return device

# ----- Δημιουργία παραθύρων -----
def create_sequences(X, y, lookback):
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    Xs, ys = [], [] 
    for i in range(len(X) - lookback):
    # Check if this window belongs to a single group
        Xs.append(X[i:i+lookback])
        #ys.append(y[i+lookback])
        ys.append(y[i+1:i+1+lookback])
    return np.stack(Xs), np.array(ys)

def circular_shift_df(dt):
    dt_ = pd.DataFrame(dt)
    dt_ = dt_.apply(lambda spiketrain: np.roll(spiketrain,
    np.random.randint(1, spiketrain.shape[0] - 1)))
    return dt_.to_numpy()


def train_one_epoch(epoch, loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)#.view_as(logits))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1} ▶ Average Train Loss: {avg:.5f}")
    return avg


def val_test_one_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb)#.view_as(logits)).item()
    avg = total_loss / len(loader)
    print(f"  Val Loss: {avg:.5f}")
    return avg
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def compute_metrics(y_true, y_pred, y_probs):
    # Flatten για να τα κάνουμε 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_probs = y_probs.flatten()
    # threshold‐dependent
    acc  = accuracy_score(y_true, y_pred)            # no zero_division here
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score   (y_true, y_pred, zero_division=0)
    f1   = f1_score       (y_true, y_pred, zero_division=0)
    # threshold‐independent
    ap   = average_precision_score(y_true, y_probs)
    auc  = roc_auc_score(y_true, y_probs)
    # specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp + 1e-8)
    return acc, prec, rec, spec, f1, ap, auc


def get_logits_probs(ys_train, ys_val, ys_test, train_loader, val_loader, test_loader, model, device):
    ys_train_true = ys_train
    ys_val_true = ys_val
    ys_test_true = ys_test

    # 1) Collect raw logits
    ys_train_logits = []
    ys_val_logits   = []
    ys_test_logits  = []

    model.eval()
    with torch.no_grad():
        for xb, _ in train_loader:
            ys_train_logits.append(model(xb.to(device)).cpu())
        for xb, _ in val_loader:
            ys_val_logits.append(  model(xb.to(device)).cpu())
        for xb, _ in test_loader:
            ys_test_logits.append( model(xb.to(device)).cpu())

    # 2) Stack into single tensors/arrays
    ys_train_logits = torch.cat(ys_train_logits, dim=0).numpy()#.flatten()
    ys_val_logits   = torch.cat(ys_val_logits,   dim=0).numpy()#.flatten()
    ys_test_logits  = torch.cat(ys_test_logits,  dim=0).numpy()#.flatten()


    # 4) Convert to probabilities
    ys_train_probs = sigmoid(ys_train_logits)
    ys_val_probs   = sigmoid(ys_val_logits)
    ys_test_probs  = sigmoid(ys_test_logits)

    return ys_train_logits, ys_val_logits, ys_test_logits, ys_train_probs, ys_val_probs, ys_test_probs#, ys_train_pred, ys_val_pred, ys_test_pred

def get_threshold(ys_val_true, ys_val_probs):
        # >>threshold<<
    prec, rec, th = precision_recall_curve(ys_val_true, ys_val_probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx   = f1_scores.argmax()
    best_threshold = th[best_idx]
    ld = th[best_idx]
    #print("Best Val F1 threshold:", best_threshold, "→ F1:", f1_scores[best_idx])

    # # βάλε το threshold σου στο επόμενο βήμα
    threshold = best_threshold
        
    # threshold = 0.5
    return threshold
def get_stat_threshold(eventograms_L23, neuron):
    count_ones = eventograms_L23[[neuron]].sum().values[0]
    num_of_frames = len(eventograms_L23)
    threshold = count_ones / num_of_frames
    return threshold

def get_preds(ys_train_probs, ys_val_probs, ys_test_probs, threshold):
    # (Optional) 5) Hard 0/1 preds at threshold 0.5
    ys_train_pred = (ys_train_probs > threshold).astype(int)
    ys_val_pred   = (ys_val_probs   > threshold).astype(int)
    ys_test_pred  = (ys_test_probs  > threshold).astype(int)
    return ys_train_pred, ys_val_pred, ys_test_pred

def create_datasets(Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test, Xs_null_train, Xs_null_val, Xs_null_test):
    train_dataset = TensorDataset(Xs_train, ys_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    val_dataset = TensorDataset(Xs_val, ys_val)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    test_dataset = TensorDataset(Xs_test, ys_test)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    train_dataset_null = TensorDataset(Xs_null_train, ys_train)
    train_loader_null = DataLoader(train_dataset_null, batch_size, shuffle=True)

    val_dataset_null = TensorDataset(Xs_null_val, ys_val)
    val_loader_null = DataLoader(val_dataset_null, batch_size, shuffle=False)

    test_dataset_null = TensorDataset(Xs_null_test, ys_test)
    test_loader_null = DataLoader(test_dataset_null, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_loader_null, val_loader_null, test_loader_null

def sequences_to_tensors(Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test, Xs_null_train, Xs_null_val, Xs_null_test):
    Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
    ys_train = torch.tensor(ys_train, dtype=torch.float32)#.unsqueeze(1)

    Xs_val = torch.tensor(Xs_val, dtype=torch.float32)
    ys_val = torch.tensor(ys_val, dtype=torch.float32)#.unsqueeze(1)

    Xs_test = torch.tensor(Xs_test, dtype=torch.float32)
    ys_test = torch.tensor(ys_test, dtype=torch.float32)#.unsqueeze(1)

    Xs_null_train = torch.tensor(Xs_null_train, dtype=torch.float32)

    Xs_null_val = torch.tensor(Xs_null_val, dtype=torch.float32)

    Xs_null_test = torch.tensor(Xs_null_test, dtype=torch.float32)
    return Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test, Xs_null_train, Xs_null_val, Xs_null_test

def plot_learning_curves(train_losses, val_losses, final_test_loss,
                         null_train_losses, null_val_losses, null_final_test_loss, out_dir):
    return
    # Original palette (blues)
    orig_colors = sns.color_palette("Blues", 3)

    # Null palette: sample 5 from YlOrBr, then take the first 3 (yellow→light orange)
    null_colors = sns.color_palette("YlOrBr", 5)[:3]

    plt.figure(figsize=(12, 6))

    # ─── Original model (all solid) ───
    plt.plot(train_losses, label='Train', color=orig_colors[0], linewidth=2, linestyle='-')
    plt.plot(val_losses,   label='Val',   color=orig_colors[1], linewidth=2, linestyle='-')
    plt.hlines(final_test_loss, 0, num_epochs-1,
            label=f'Test = {final_test_loss:.3f}',
            colors=[orig_colors[2]], linestyles='-', linewidth=2)

    # ─── Null model (all solid) ───
    plt.plot(null_train_losses, label='Null Train', color=null_colors[0], linewidth=2, linestyle='-')
    plt.plot(null_val_losses,   label='Null Val',   color=null_colors[1], linewidth=2, linestyle='-')
    plt.hlines(null_final_test_loss, 0, num_epochs-1,
            label=f'Null Test = {null_final_test_loss:.3f}',
            colors=[null_colors[2]], linestyles='-', linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss",  fontsize=14)
    plt.title("Learning Curves: Original vs. Null Models")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(out_dir, "average_loss.png"))
    plt.close()


def keep_metrics(ys_train, ys_val, ys_test, train_loader, val_loader, test_loader,
                 ys_train_null, ys_val_null, ys_test_null, train_loader_null, val_loader_null, test_loader_null, out_dir, run_name, model, device, threshold):
    ys_train_true = ys_train
    ys_val_true = ys_val
    ys_test_true = ys_test

    ys_train_logits, ys_val_logits, ys_test_logits, ys_train_probs, ys_val_probs, ys_test_probs = get_logits_probs(ys_train, ys_val, ys_test, train_loader, val_loader, test_loader, model, device)
    ys_train_logits_null, ys_val_logits_null, ys_test_logits_null, ys_train_probs_null, ys_val_probs_null, ys_test_probs_null = get_logits_probs(ys_train, ys_val, ys_test, train_loader_null, val_loader_null, test_loader_null, model, device)

     #get_threshold(ys_val_true, ys_val_probs)

    ys_train_pred, ys_val_pred, ys_test_pred = get_preds(ys_train_probs, ys_val_probs, ys_test_probs, threshold)
    ys_train_pred_null, ys_val_pred_null, ys_test_pred_null = get_preds(ys_train_probs_null, ys_val_probs_null, ys_test_probs_null, threshold)


    train_metrics      = compute_metrics(ys_train_true,      ys_train_pred,      ys_train_probs)
    val_metrics        = compute_metrics(ys_val_true,        ys_val_pred,        ys_val_probs)
    test_metrics       = compute_metrics(ys_test_true,       ys_test_pred,       ys_test_probs)
    null_train_metrics = compute_metrics(ys_train_true, ys_train_pred_null, ys_train_probs_null)
    null_val_metrics   = compute_metrics(ys_val_true,   ys_val_pred_null,   ys_val_probs_null)
    null_test_metrics  = compute_metrics(ys_test_true,  ys_test_pred_null,  ys_test_probs_null)

    metrics_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AP", "ROC AUC"]

    train_vals      = train_metrics
    val_vals        = val_metrics
    test_vals       = test_metrics
    null_train_vals = null_train_metrics
    null_val_vals   = null_val_metrics
    null_test_vals  = null_test_metrics

    x     = np.arange(len(metrics_names))
    width = 0.15


    df_metrics = pd.DataFrame({
    "split":      ["Train","Val","Test","Null Train","Null Val","Null Test"],
            "Accuracy":   [*train_metrics[0:1], *val_metrics[0:1], *test_metrics[0:1],
                        *null_train_metrics[0:1], *null_val_metrics[0:1], *null_test_metrics[0:1]],
            "Precision":  [train_metrics[1], val_metrics[1], test_metrics[1],
                        null_train_metrics[1], null_val_metrics[1], null_test_metrics[1]],
            "Recall":     [train_metrics[2], val_metrics[2], test_metrics[2],
                        null_train_metrics[2], null_val_metrics[2], null_test_metrics[2]],
            "Specificity":[train_metrics[3], val_metrics[3], test_metrics[3],
                        null_train_metrics[3], null_val_metrics[3], null_test_metrics[3]],
            "F1":         [train_metrics[4], val_metrics[4], test_metrics[4],
                        null_train_metrics[4], null_val_metrics[4], null_test_metrics[4]],
            "AP":         [train_metrics[5], val_metrics[5], test_metrics[5],
                        null_train_metrics[5], null_val_metrics[5], null_test_metrics[5]],
            "ROC AUC":    [train_metrics[6], val_metrics[6], test_metrics[6],
                        null_train_metrics[6], null_val_metrics[6], null_test_metrics[6]],
        })
    df_metrics.to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False)
    # 1) Read or initialize the master summary file
    summary_path = os.path.join(args.out_root, "all_runs_summary.csv")
    if not os.path.exists(summary_path):
        # write header
        with open(summary_path, "w") as f:
            f.write("run,split,Accuracy,Precision,Recall,Specificity,F1,AP,ROC AUC\n")

    # 2) Append this run’s metrics_table.csv into the master summary
    metrics_table = pd.read_csv(os.path.join(out_dir, "metrics_table.csv"))
    # prepend a column for the run name
    metrics_table.insert(0, "run", run_name)
    # append to all_runs_summary.csv
    metrics_table.to_csv(summary_path, mode="a", header=False, index=False)
    print(f"Finished {run_name}")

#>>loop function
def train_and_evaluate(eventograms_L23, 
                       eventograms_L4, 
                       neuron_groups_dict, 
                       hidden_size, 
                       lookback, 
                       neuron, 
                       num_epochs, 
                       learning_rate, 
                       device, 
                       out_root, 
                       output_size,
                       threshold,
                       patience=5): # run_counter, total_runs, out_root):

    print(f"Running test: "
          f"hidden_size={hidden_size}, "
            f"lookback={lookback}, "
            f"neuron={neuron}, "
            f"epochs={num_epochs}, "
            f"lr={learning_rate}"
        )

    run_name = (f"hs{hidden_size}_lb{lookback}_"
                f"{neuron}_ep{num_epochs}_lr{learning_rate}")
    out_dir  = os.path.join(out_root, run_name)

    # —————————— Νεος κώδικας για καθαρό φάκελο ——————————
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    # count ones in all of eventograms_L4_15_dc_data_mouse3[[neuron]]
    
    count_ones = eventograms_L23[[neuron]].sum().values[0]

    print(f"Count of ones in {neuron}: {count_ones} / {len(eventograms_L23)} or {count_ones / len(eventograms_L23) * 100:.2f}%")

    #new
    nid = int(neuron.lstrip("V"))
    l4_ids = neuron_groups_dict.get(nid, [])
    l4_cols = [f"V{lid}" for lid in l4_ids if f"V{lid}" in eventograms_L4.columns]
    print(f"[{neuron}] using {len(l4_cols)} L4 cols")
    #

    df = pd.concat([ eventograms_L23[[neuron]], eventograms_L4[l4_cols] ], axis=1)

    print(df.shape)  # should be (23070, 1193) for L23 and should be (23070, 2670) for L4
    X = df
    y = df[neuron]

    X_null = circular_shift_df(X)

    train_frac = 0.80
    val_frac   = 0.15
    #test_frac = 0.05
    test_frac  = 1 - train_frac - val_frac  # = 0.05

    N = len(X)
    train_end = int(N * train_frac)               # π.χ. 0.80*N
    val_end   = int(N * (train_frac + val_frac))  # π.χ. 0.95*N

    X_train = X[:train_end]
    X_val   = X[train_end:val_end]
    X_test  = X[val_end:]

    y_train = y[:train_end]
    y_val   = y[train_end:val_end]
    y_test  = y[val_end:]

    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape:   {X_val.shape},   y_val.shape:   {y_val.shape}")
    print(f"X_test.shape:  {X_test.shape},  y_test.shape:  {y_test.shape}")

    Xs_train, ys_train = create_sequences(X_train, y_train, lookback)
    Xs_val, ys_val = create_sequences(X_val, y_val, lookback)
    Xs_test, ys_test = create_sequences(X_test, y_test, lookback)
    #(for Xs_train.shape 23070 is all frames we use train_fac * 23070 so 18456 frames as training)


    print(f"Xs_train.shape: {Xs_train.shape}, ys_train.shape: {ys_train.shape}")
    print(f"Xs_val.shape:   {Xs_val.shape},   ys_val.shape:   {ys_val.shape}")
    print(f"Xs_test.shape:  {Xs_test.shape},  ys_test.shape:  {ys_test.shape}")

    X_null_train = X_null[:train_end]
    X_null_val   = X_null[train_end:val_end]
    X_null_test  = X_null[val_end:]


    Xs_null_train, ys_train = create_sequences(X_null_train, y_train, lookback)
    Xs_null_val, ys_val = create_sequences(X_null_val, y_val, lookback)
    Xs_null_test, ys_test = create_sequences(X_null_test, y_test, lookback)


    Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test, Xs_null_train, Xs_null_val, Xs_null_test = sequences_to_tensors(
        Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test, Xs_null_train, Xs_null_val, Xs_null_test
    )   

    print(f"Xs_train.shape: {Xs_train.shape}, ys_train.shape: {ys_train.shape}")    
    print(f"Xs_val.shape:   {Xs_val.shape},   ys_val.shape:   {ys_val.shape}")
    print(f"Xs_test.shape:  {Xs_test.shape},  ys_test.shape:  {ys_test.shape}")


    train_loader, val_loader, test_loader, train_loader_null, val_loader_null, test_loader_null = create_datasets(
        Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test, Xs_null_train, Xs_null_val, Xs_null_test)
    

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break  # Just show the first batch

    input_size = X.shape[1]
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Lookback: {lookback}")
   
    class LSTMNetwork(nn.Module):
        def __init__(self, input_size=input_size, hidden_size= hidden_size, num_layers=num_layers, output_size=output_size):
            super(LSTMNetwork, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc   = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            #out = out[:, -1, :]
            logits = self.fc(out)
            return logits.squeeze(-1)               # raw scores (“logits”)
    model = LSTMNetwork().to(device)
    torch.cuda.empty_cache()

    #criterion = nn.BCEWithLogitsLoss() # internally applies sigmoid + BCELoss

    # eps = 1e-5
    # # num_of_frames - count_ones gives negatives; count_ones + eps avoids division by zero
    # pos_weights = torch.tensor(
    #     (num_of_frames - count_ones) / (count_ones + eps),
    #     dtype=torch.float32,
    #     device=device
    # )

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)


    criterion = nn.CrossEntropyLoss() # for multi-class classification

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    display(model)  



    # >>run

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
    #     train_losses.append(train_one_epoch(epoch, train_loader, model, optimizer, criterion, device))
    #     val_losses.append(val_test_one_epoch(val_loader, model, criterion, device))
    # final_test_loss = val_test_one_epoch(test_loader, model, criterion, device)
        # 1) Train
        train_loss = train_one_epoch(epoch, train_loader, model, optimizer, criterion, device)
        train_losses.append(train_loss)

        # 2) Validate
        val_loss = val_test_one_epoch(val_loader, model, criterion, device)
        val_losses.append(val_loss)

        # 3) Check early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # (Optional) Save a checkpoint of the best model
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.5f}")
            break

    # 4) After stopping, you can load the best model before final test:
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pt')))
    final_test_loss = val_test_one_epoch(test_loader, model, criterion, device)


    null_train_losses, null_val_losses = [], []
    best_null_val_loss   = float('inf')
    epochs_no_improve_null = 0
    for epoch in range(num_epochs):
               # 1) Train on null data
        null_train_loss = train_one_epoch(epoch, train_loader_null, model, optimizer, criterion, device)
        null_train_losses.append(null_train_loss)

        # 2) Validate on null data
        null_val_loss = val_test_one_epoch(val_loader_null, model, criterion, device)
        null_val_losses.append(null_val_loss)

        # 3) Early‐Stopping check
        if null_val_loss < best_null_val_loss:
            best_null_val_loss = null_val_loss
            epochs_no_improve_null = 0
            # save your null‐model checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, 'best_model_null.pt')
            )
        else:
            epochs_no_improve_null += 1
            print(f"  ↳ Null: no improvement for {epochs_no_improve_null} epoch(s).")

        if epochs_no_improve_null >= patience:
            print(
                f"Null‐model early stopping at epoch {epoch+1}. "
                f"Best null val loss: {best_null_val_loss:.5f}"
            )
            break

    # 4) Load the best null‐model before final test
    model.load_state_dict(
        torch.load(os.path.join(out_dir, 'best_model_null.pt'))
    )
    null_final_test_loss = val_test_one_epoch(test_loader_null, model, criterion, device)

    #>>stop
    plot_learning_curves(train_losses, val_losses, final_test_loss,
                         null_train_losses, null_val_losses, null_final_test_loss, out_dir)                         
    return keep_metrics(
        ys_train, ys_val, ys_test, train_loader, val_loader, test_loader,
        ys_train, ys_val, ys_test, train_loader_null, val_loader_null, test_loader_null,
        out_dir, run_name, model, device, threshold
    )



#main
#load data
within_boundaries_data_mouse3 = pd.read_csv("../data/mouse24617_withinBoundaries_15um-no_gap.csv")

L4_neurons_per_Layer_data_mouse3 = pd.read_csv("../data/mouse24617_L4_neuronPerLayer_V1_0.01Hz.csv")

temp_set1 = set(within_boundaries_data_mouse3["x"])
temp_set2 = set(L4_neurons_per_Layer_data_mouse3["Neurons"])

temp_intersection1 = temp_set1.intersection(temp_set2)

L4_neurons_per_Layer_within_boundaries_data_mouse3 = pd.DataFrame(sorted(temp_intersection1), columns=["Neuron_IDs"])


L23_neurons_per_Layer_data_mouse3 = pd.read_csv("../data/mouse24617_L23_neuronPerLayer_V1_0.01Hz.csv")

temp_set1 = set(within_boundaries_data_mouse3.iloc[:, 0])
temp_set2 = set(L23_neurons_per_Layer_data_mouse3.iloc[:, 0])

temp_intersection2 = temp_set1.intersection(temp_set2)

L23_neurons_per_Layer_within_boundaries_data_mouse3 = pd.DataFrame(sorted(temp_intersection2), columns=["Neuron_IDs"])


L234_neurons_within_boundaries_data_mouse3 = pd.concat(
    [L4_neurons_per_Layer_within_boundaries_data_mouse3,
     L23_neurons_per_Layer_within_boundaries_data_mouse3],
    ignore_index=True
)
L234_neurons_within_boundaries_data_mouse3.columns = ["Neuron_IDs"]


l4_ids  = set(L4_neurons_per_Layer_within_boundaries_data_mouse3["Neuron_IDs"])
l23_ids = set(L23_neurons_per_Layer_within_boundaries_data_mouse3["Neuron_IDs"])


STTC_500shifts_0_dt_data_mouse3 = pd.read_csv("../data/mouse24617_STTC_3nz_1.5dc_full_60min_500-shifts_0-dt_pairs.csv")

# make a set of the “within‐boundaries” IDs
allowed_ids = set(L234_neurons_within_boundaries_data_mouse3["Neuron_IDs"])

# filter the STTC pairs so both ends are in that set
STTC_L234_500shifts_0_dt_data_mouse3 = (
    STTC_500shifts_0_dt_data_mouse3[
        STTC_500shifts_0_dt_data_mouse3["NeuronA"].isin(allowed_ids)
        & STTC_500shifts_0_dt_data_mouse3["NeuronB"].isin(allowed_ids)
    ]
    .copy()
)

# compute z-score and build new DataFrame
z_score_L234_500shifts_0_dt_data_mouse3 = (STTC_L234_500shifts_0_dt_data_mouse3
    .assign(z_score=(STTC_L234_500shifts_0_dt_data_mouse3['STTC'] - STTC_L234_500shifts_0_dt_data_mouse3['CtrlGrpMean']) / STTC_L234_500shifts_0_dt_data_mouse3['CtrlGrpStDev'])
    [['NeuronA', 'NeuronB', 'z_score']]
)

z_score_L234_500shifts_0_dt_data_mouse3_more_than_4 = z_score_L234_500shifts_0_dt_data_mouse3[z_score_L234_500shifts_0_dt_data_mouse3['z_score'] >= 4]

z_score_L234_500shifts_0_dt_data_mouse3_more_than_4 = z_score_L234_500shifts_0_dt_data_mouse3_more_than_4.reset_index(drop=True)

z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A = z_score_L234_500shifts_0_dt_data_mouse3_more_than_4[z_score_L234_500shifts_0_dt_data_mouse3_more_than_4['NeuronA'].isin(l23_ids)]

z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A_L4B = z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A[z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A['NeuronB'].isin(l4_ids)]

z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A_L4B = z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A_L4B.reset_index(drop=True)


neuron_groups_dict = defaultdict(set)
for a, b in zip(z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A_L4B['NeuronA'], z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A_L4B['NeuronB']):
    neuron_groups_dict[a].add(b)

# και αν θες κανονικό dict
neuron_groups_dict = dict(neuron_groups_dict)

l4_ids_groups = z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A_L4B['NeuronB'].unique().tolist() 
l23_ids_groups = z_score_L234_500shifts_0_dt_data_mouse3_more_than_4_l23A_L4B['NeuronA'].unique().tolist()


eventograms_15_dc_data_mouse3 = pd.read_csv("../data/mouse24617_IoannisThreshold_3nz_1.5dc_full_60min.csv")


allowed = list(set(l4_ids_groups).union(set(l23_ids_groups)))

cols = [f"V{nid}" for nid in allowed if f"V{nid}" in eventograms_15_dc_data_mouse3.columns]

eventograms_L234_15_dc_data_mouse3 = (
    eventograms_15_dc_data_mouse3[cols]
    .copy()
)

l4_cols  = [f"V{nid}" for nid in l4_ids  if f"V{nid}" in eventograms_L234_15_dc_data_mouse3.columns]
l23_cols = [f"V{nid}" for nid in l23_ids if f"V{nid}" in eventograms_L234_15_dc_data_mouse3.columns]

eventograms_L4_15_dc_data_mouse3  = eventograms_L234_15_dc_data_mouse3[l4_cols].copy()
eventograms_L23_15_dc_data_mouse3 = eventograms_L234_15_dc_data_mouse3[l23_cols].copy()

print(f"eventograms_L4_15_dc_data_mouse3.shape: {eventograms_L4_15_dc_data_mouse3.shape}"   
      f"eventograms_L23_15_dc_data_mouse3.shape: {eventograms_L23_15_dc_data_mouse3.shape}"     )


# Setting up the device for PyTorch
device = my_cuda()


#>>parameters<<<
np.random.seed(42)
frame_start_mouse3 = 26919
frame_end_mouse3 = 49988
time_in_sec_mouse3 = 3661
num_of_frames_mouse3 = frame_end_mouse3 - frame_start_mouse3 + 1
ms_per_frame_mouse3 = time_in_sec_mouse3 * 1000 / (num_of_frames_mouse3)   
#>>batch
l4_ids =l4_ids_groups
l23_ids = l23_ids_groups
num_of_neurons_l4 = len(l4_ids)
num_of_neurons_l23 = len(l23_ids)
num_of_frames = num_of_frames_mouse3
batch_size = 32
num_layers = 1
# Ορίζουμε πόσοι workers
NUM_WORKERS = max(os.cpu_count() - 1, 1) 
num_epochs = 100
output_size = 3  # since we are predicting a single neuron activity
# #----------------------
# lookback = 5
# neuron = "V8213"
# learing_rate = 0.001
# hidden_size = 5

# num_epochs = 100
# #--------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[10])
parser.add_argument("--lookbacks",    nargs="+", type=int, default=[10])
# parser.add_argument("--neurons",      nargs="+", type=str,default=["V8192"])
# parser.add_argument("--epochs",       nargs="+", type=int, default=[8])
parser.add_argument("--lr",           nargs="+", type=float, default=[0.001])
# parser.add_argument("--batch_sizes",   type=int,   default=1024)
parser.add_argument("--out_root",     type=str,   default="results")
args = parser.parse_args()

#loop
# for hidden_size in args.hidden_sizes:
#     for lookback in args.lookbacks:
#         for learning_rate in args.lr:
#             desc = (f"hs={hidden_size} lb={lookback} "
#                     f"ep={num_epochs} lr={learning_rate}")
#             train_and_evaluate(
#                 eventograms_L23 = eventograms_L23_15_dc_data_mouse3,
#                 eventograms_L4  = eventograms_L4_15_dc_data_mouse3,
#                 neuron_groups_dict = neuron_groups_dict,
#                 hidden_size     = hidden_size,
#                 lookback        = lookback,
#                 neuron          = "V368",  # or any other neuron you want to test
#                 num_epochs      = num_epochs,
#                 learning_rate   = learning_rate,
#                 device          = device,
#                 out_root        = args.out_root
#             )   




# multithreding code 



#loop
for hidden_size in args.hidden_sizes:
    for lookback in args.lookbacks:
        for learning_rate in args.lr:
            desc = (f"hs={hidden_size} lb={lookback} "
                    f"ep={num_epochs} lr={learning_rate}")
            
            # φτιάχνουμε τα partials για κάθε νευρώνα
            jobs = []
            for nid in l23_ids_groups:
                neuron = f"V{nid}"
                job = partial(
                    train_and_evaluate,
                    eventograms_L23 = eventograms_L23_15_dc_data_mouse3,
                    eventograms_L4  = eventograms_L4_15_dc_data_mouse3,
                    neuron_groups_dict = neuron_groups_dict,
                    hidden_size     = hidden_size,
                    lookback        = lookback,
                    neuron          = neuron,
                    num_epochs      = num_epochs,
                    learning_rate   = learning_rate,
                    device          = device,
                    out_root        = args.out_root,
                    output_size     = output_size,
                    threshold         = get_stat_threshold(eventograms_L23_15_dc_data_mouse3, neuron)
                )
                jobs.append(job)

            # parallel map με tqdm
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # executor.map θα τρέξει τα job() για κάθε νευρώνα
                for _ in tqdm(executor.map(lambda fn: fn(), jobs),
                                total=len(jobs),
                                desc=desc,
                                unit="neuron"):
                    pass



