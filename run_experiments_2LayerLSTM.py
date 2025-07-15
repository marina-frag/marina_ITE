# cmd.exe /c run_experiments.bat
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
        ys.append(y[i+lookback])
        #Ys.append(Y[i+1:i+1+window_size])
    return np.stack(Xs), np.array(ys)

def circular_shift_df(dt):
    dt_ = pd.DataFrame(dt)
    dt_ = dt_.apply(lambda spiketrain: np.roll(spiketrain,
    np.random.randint(1, spiketrain.shape[0] - 1)))
    return dt_.to_numpy()


def train_one_epoch(epoch, loader):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb.view_as(logits))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1} ▶ Average Train Loss: {avg:.5f}")
    return avg


def val_test_one_epoch(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb.view_as(logits)).item()
    avg = total_loss / len(loader)
    print(f"  Val Loss: {avg:.5f}")
    return avg
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def compute_metrics(y_true, y_pred, y_probs):
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


def logits_probs_preds(ys_train, ys_val, ys_test, train_loader, val_loader, test_loader):
    
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
    ys_train_logits = torch.cat(ys_train_logits, dim=0).numpy().flatten()
    ys_val_logits   = torch.cat(ys_val_logits,   dim=0).numpy().flatten()
    ys_test_logits  = torch.cat(ys_test_logits,  dim=0).numpy().flatten()


    # 4) Convert to probabilities
    ys_train_probs = sigmoid(ys_train_logits)
    ys_val_probs   = sigmoid(ys_val_logits)
    ys_test_probs  = sigmoid(ys_test_logits)


    # >>threshold<<
    prec, rec, th = precision_recall_curve(ys_val_true, ys_val_probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx   = f1_scores.argmax()
    best_threshold = th[best_idx]


    ld = th[best_idx]
    #print("Best Val F1 threshold:", best_threshold, "→ F1:", f1_scores[best_idx])

    # # βάλε το threshold σου στο επόμενο βήμα
    # threshold = best_threshold
        
    threshold = 0.5

    # (Optional) 5) Hard 0/1 preds at threshold 0.5
    ys_train_pred = (ys_train_probs > threshold).astype(int)
    ys_val_pred   = (ys_val_probs   > threshold).astype(int)
    ys_test_pred  = (ys_test_probs  > threshold).astype(int)

    return ys_train_logits, ys_val_logits, ys_test_logits, ys_train_probs, ys_val_probs, ys_test_probs, ys_train_pred, ys_val_pred, ys_test_pred


within_boundaries_data_mouse3 = pd.read_csv(r"C:\Users\marin\Documents\ITE\PROJECT\data\mouse24617_withinBoundaries_15um-no_gap.csv")

L4_neurons_per_Layer_data_mouse3 = pd.read_csv(r"C:\Users\marin\Documents\ITE\PROJECT\data\mouse24617_L4_neuronPerLayer_V1_0.01Hz.csv")

temp_set1 = set(within_boundaries_data_mouse3["x"])
temp_set2 = set(L4_neurons_per_Layer_data_mouse3["Neurons"])

temp_intersection1 = temp_set1.intersection(temp_set2)

L4_neurons_per_Layer_within_boundaries_data_mouse3 = pd.DataFrame(sorted(temp_intersection1), columns=["Neuron_IDs"])


L23_neurons_per_Layer_data_mouse3 = pd.read_csv(r"C:\Users\marin\Documents\ITE\PROJECT\data\mouse24617_L23_neuronPerLayer_V1_0.01Hz.csv")

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

eventograms_15_dc_data_mouse3 = pd.read_csv(r"C:\Users\marin\Documents\ITE\PROJECT\data\mouse24617_IoannisThreshold_3nz_1.5dc_full_60min.csv")


allowed = set(L234_neurons_within_boundaries_data_mouse3["Neuron_IDs"])

cols = [f"V{nid}" for nid in allowed if f"V{nid}" in eventograms_15_dc_data_mouse3.columns]

eventograms_L234_15_dc_data_mouse3 = (
    eventograms_15_dc_data_mouse3[cols]
    .copy()
)

l4_ids  = set(L4_neurons_per_Layer_within_boundaries_data_mouse3["Neuron_IDs"])
l23_ids = set(L23_neurons_per_Layer_within_boundaries_data_mouse3["Neuron_IDs"])

l4_cols  = [f"V{nid}" for nid in l4_ids  if f"V{nid}" in eventograms_L234_15_dc_data_mouse3.columns]
l23_cols = [f"V{nid}" for nid in l23_ids if f"V{nid}" in eventograms_L234_15_dc_data_mouse3.columns]

eventograms_L4_15_dc_data_mouse3  = eventograms_L234_15_dc_data_mouse3[l4_cols].copy()
eventograms_L23_15_dc_data_mouse3 = eventograms_L234_15_dc_data_mouse3[l23_cols].copy()


# Setting up the device for PyTorch
device = my_cuda()

num_of_neurons_l4 = len(l4_ids)
num_of_neurons_l23 = len(l23_ids)


#>>parameters<<<
np.random.seed(42)
frame_start_mouse3 = 26919
frame_end_mouse3 = 49988
time_in_sec_mouse3 = 3661
num_of_frames_mouse3 = frame_end_mouse3 - frame_start_mouse3 + 1
ms_per_frame_mouse3 = time_in_sec_mouse3 * 1000 / (num_of_frames_mouse3)   

num_of_neurons_l4 = len(l4_ids)
num_of_neurons_l23 = len(l23_ids)
num_of_frames = num_of_frames_mouse3
batch_size = 1024
num_layers = 2
# #----------------------
# lookback = 5
# neuron = "V8213"
# learing_rate = 0.001
# hidden_size = 5

# num_epochs = 100
# #--------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[8])
parser.add_argument("--lookbacks",    nargs="+", type=int, default=[1])
# parser.add_argument("--neurons",      nargs="+", type=str,default=["V8192"])
parser.add_argument("--epochs",       nargs="+", type=int, default=[50])
parser.add_argument("--lr",           nargs="+", type=float, default=[0.001])
# parser.add_argument("--batch_sizes",   type=int,   default=1024)
parser.add_argument("--out_root",     type=str,   default="results")
args = parser.parse_args()

# ─── prepare run counter ─────────────────────────────────────────────────────────
total_runs = (
    len(args.hidden_sizes)
  * len(args.lookbacks)
  * len(l23_ids)
  * len(args.epochs)
  * len(args.lr)
)
run_counter = 0
#>>loop

# ─── 3) Sweep over hyper-params ────────────────────────────────────────────────
for hidden_size in args.hidden_sizes:
    for lookback in args.lookbacks:
        for nid in l23_ids:
            neuron = "V" + str(nid)
            for num_epochs in args.epochs:
                for learning_rate in args.lr:

                    run_counter += 1
                    print(
                        f"Running test {run_counter}/{total_runs}: "
                        f"hidden_size={hidden_size}, "
                        f"lookback={lookback}, "
                        f"neuron={neuron}, "
                        f"epochs={num_epochs}, "
                        f"lr={learning_rate}"
                    )

                    run_name = (f"hs{hidden_size}_lb{lookback}_"
                                f"{neuron}_ep{num_epochs}_lr{learning_rate}")
                    out_dir  = os.path.join(args.out_root, run_name)

                    # —————————— Νεος κώδικας για καθαρό φάκελο ——————————
                    if os.path.isdir(out_dir):
                        shutil.rmtree(out_dir)
                    os.makedirs(out_dir)
                    # count ones in all of eventograms_L4_15_dc_data_mouse3[[neuron]]
                    count_ones = eventograms_L23_15_dc_data_mouse3[[neuron]].sum().values[0]
                    print(f"Count of ones in {neuron}: {count_ones} / {len(eventograms_L23_15_dc_data_mouse3)} or {count_ones / len(eventograms_L23_15_dc_data_mouse3) * 100:.2f}%")

                    df = pd.concat([ eventograms_L23_15_dc_data_mouse3[[neuron]], eventograms_L4_15_dc_data_mouse3], axis=1)
                    print(df.shape)  # should be (23070, 1193) for L23 and should be (23070, 2670) for L4
                    #display(df)

                    #from sklearn.model_selection import train_test_split
                    #openpyxl


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

                    # print(X_train.shape, X_val.shape, X_test.shape,
                    #       y_train.shape, y_val.shape, y_test.shape)

                    Xs_train, ys_train = create_sequences(X_train, y_train, lookback)
                    Xs_val, ys_val = create_sequences(X_val, y_val, lookback)
                    Xs_test, ys_test = create_sequences(X_test, y_test, lookback)
                    #(for Xs_train.shape 23070 is all frames we use train_fac * 23070 so 18456 frames as training)

                    X_null_train = X_null[:train_end]
                    X_null_val   = X_null[train_end:val_end]
                    X_null_test  = X_null[val_end:]


                    Xs_null_train, ys_train = create_sequences(X_null_train, y_train, lookback)
                    Xs_null_val, ys_val = create_sequences(X_null_val, y_val, lookback)
                    Xs_null_test, ys_test = create_sequences(X_null_test, y_test, lookback)


                    Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
                    ys_train = torch.tensor(ys_train, dtype=torch.float32).unsqueeze(1)

                    Xs_val = torch.tensor(Xs_val, dtype=torch.float32)
                    ys_val = torch.tensor(ys_val, dtype=torch.float32).unsqueeze(1)

                    Xs_test = torch.tensor(Xs_test, dtype=torch.float32)
                    ys_test = torch.tensor(ys_test, dtype=torch.float32).unsqueeze(1)

                    Xs_null_train = torch.tensor(Xs_null_train, dtype=torch.float32)

                    Xs_null_val = torch.tensor(Xs_null_val, dtype=torch.float32)

                    Xs_null_test = torch.tensor(Xs_null_test, dtype=torch.float32)


                    train_dataset = TensorDataset(Xs_train, ys_train)
                    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

                    val_dataset = TensorDataset(Xs_val, ys_val)
                    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

                    test_dataset = TensorDataset(Xs_test, ys_test)
                    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

                    for _, batch in enumerate(train_loader):
                        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                        print(x_batch.shape, y_batch.shape)
                        break  # Just show the first batch

                    class LSTMNetwork(nn.Module):
                        def __init__(self, input_size=num_of_neurons_l4+1, hidden_size= hidden_size, num_layers=num_layers, output_size=1):
                            super(LSTMNetwork, self).__init__()
                            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                            self.fc   = nn.Linear(hidden_size, output_size)

                        def forward(self, x):
                            out, _ = self.lstm(x)
                            out = out[:, -1, :]
                            logits = self.fc(out)
                            return logits               # raw scores (“logits”)
                    model = LSTMNetwork().to(device)

                    #criterion = nn.BCEWithLogitsLoss() # internally applies sigmoid + BCELoss

                    eps = 1e-5
                    # num_of_frames - count_ones gives negatives; count_ones + eps avoids division by zero
                    pos_weights = torch.tensor(
                        (num_of_frames - count_ones) / (count_ones + eps),
                        dtype=torch.float32,
                        device=device
                    )

                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)


                    #criterion = nn.MSELoss()

                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    display(model)  



                    # >>run

                    train_losses, val_losses = [], []
                    for epoch in range(num_epochs):
                        train_losses.append(train_one_epoch(epoch, train_loader))
                        val_losses.append(val_test_one_epoch(val_loader))
                    final_test_loss = val_test_one_epoch(test_loader)


                    train_dataset_null = TensorDataset(Xs_null_train, ys_train)
                    train_loader_null = DataLoader(train_dataset_null, batch_size, shuffle=True)

                    val_dataset_null = TensorDataset(Xs_null_val, ys_val)
                    val_loader_null = DataLoader(val_dataset_null, batch_size, shuffle=False)

                    test_dataset_null = TensorDataset(Xs_null_test, ys_test)
                    test_loader_null = DataLoader(test_dataset_null, batch_size, shuffle=False)



                    null_train_losses, null_val_losses = [], []
                    for epoch in range(num_epochs):
                        null_train_losses.append(train_one_epoch(epoch, train_loader_null))
                        null_val_losses.append(val_test_one_epoch(val_loader_null))
                    null_final_test_loss = val_test_one_epoch(test_loader_null)

                    #>>stop

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

                    ys_train_true = ys_train
                    ys_val_true = ys_val
                    ys_test_true = ys_test

                    ys_train_logits, ys_val_logits, ys_test_logits, ys_train_probs, ys_val_probs, ys_test_probs, ys_train_pred, ys_val_pred, ys_test_pred = logits_probs_preds(ys_train, ys_val, ys_test, train_loader, val_loader, test_loader)
                    ys_train_logits_null, ys_val_logits_null, ys_test_logits_null, ys_train_probs_null, ys_val_probs_null, ys_test_probs_null, ys_train_pred_null, ys_val_pred_null, ys_test_pred_null = logits_probs_preds(ys_train, ys_val, ys_test, train_loader_null, val_loader_null, test_loader_null)

                    # >>run2

                    train_metrics      = compute_metrics(ys_train_true,      ys_train_pred,      ys_train_probs)
                    val_metrics        = compute_metrics(ys_val_true,        ys_val_pred,        ys_val_probs)
                    test_metrics       = compute_metrics(ys_test_true,       ys_test_pred,       ys_test_probs)
                    null_train_metrics = compute_metrics(ys_train_true, ys_train_pred_null, ys_train_probs_null)
                    null_val_metrics   = compute_metrics(ys_val_true,   ys_val_pred_null,   ys_val_probs_null)
                    null_test_metrics  = compute_metrics(ys_test_true,  ys_test_pred_null,  ys_test_probs_null)

                    metrics_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AP", "ROC AUC"]

                    # ─── 2) Plot Precision–Recall curves ─────────────────────────────────────────────
                    plt.figure(figsize=(12, 5))
                    real_colors = sns.color_palette("Blues", 3)
                    null_colors = sns.color_palette("YlOrBr", 3)

                    # Real model PR curves
                    for (name, y_t, y_p), c in zip([
                            ("Train", ys_train_true, ys_train_probs),
                            ("Val",   ys_val_true,   ys_val_probs),
                            ("Test",  ys_test_true,  ys_test_probs)
                        ], real_colors):
                        prec, rec, _ = precision_recall_curve(y_t, y_p)
                        ap = average_precision_score(y_t, y_p)
                        plt.plot(rec, prec, label=f"{name} (AP={ap:.2f})", color=c, linewidth=2)

                    # Null model PR curves (dashed)
                    for (name, y_t, y_p), c in zip([
                            ("Null Train", ys_train_true, ys_train_probs_null),
                            ("Null Val",   ys_val_true,   ys_val_probs_null),
                            ("Null Test",  ys_test_true,  ys_test_probs_null)
                        ], null_colors):
                        prec, rec, _ = precision_recall_curve(y_t, y_p)
                        ap = average_precision_score(y_t, y_p)
                        plt.plot(rec, prec, label=f"{name} (AP={ap:.2f})", color=c, linewidth=2, linestyle="--")

                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision–Recall Curves")
                    plt.legend(loc="lower left")
                    plt.grid(True)
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(os.path.join(out_dir, "pr_curves.png"))
                    plt.close()


                    # ─── 3) Plot ROC curves ─────────────────────────────────────────────────────────
                    plt.figure(figsize=(12, 5))

                    # Real model ROC curves
                    for (name, y_t, y_p), c in zip([
                            ("Train", ys_train_true, ys_train_probs),
                            ("Val",   ys_val_true,   ys_val_probs),
                            ("Test",  ys_test_true,  ys_test_probs)
                        ], real_colors):
                        fpr, tpr, _ = roc_curve(y_t, y_p)
                        auc = roc_auc_score(y_t, y_p)
                        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=c, linewidth=2)

                    # Null model ROC curves (dashed)
                    for (name, y_t, y_p), c in zip([
                            ("Null Train", ys_train_true, ys_train_probs_null),
                            ("Null Val",   ys_val_true,   ys_val_probs_null),
                            ("Null Test",  ys_test_true,  ys_test_probs_null)
                        ], null_colors):
                        fpr, tpr, _ = roc_curve(y_t, y_p)
                        auc = roc_auc_score(y_t, y_p)
                        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=c, linewidth=2, linestyle="--")

                    # Diagonal
                    plt.plot([0,1], [0,1], "--", color="gray", linewidth=1)

                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curves")
                    plt.legend(loc="lower right", fontsize=10)
                    plt.grid(True)
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(os.path.join(out_dir, "pr_curves.png"))
                    plt.close()


                    # ─── 4) Bar chart of scalar metrics ────────────────────────────────────────────
                    # unpack metrics tuples into lists
                    train_vals      = train_metrics
                    val_vals        = val_metrics
                    test_vals       = test_metrics
                    null_train_vals = null_train_metrics
                    null_val_vals   = null_val_metrics
                    null_test_vals  = null_test_metrics

                    x     = np.arange(len(metrics_names))
                    width = 0.15

                    plt.figure(figsize=(14, 6))
                    # real model bars
                    plt.bar(x - 1.5*width, train_vals,      width, label="Train",      color=real_colors[0])
                    plt.bar(x - 0.5*width, val_vals,        width, label="Val",        color=real_colors[1])
                    plt.bar(x + 0.5*width, test_vals,       width, label="Test",       color=real_colors[2])
                    # null model bars (with alpha)
                    plt.bar(x + 1.5*width, null_train_vals, width, label="Null Train", color=null_colors[0], alpha=0.7)
                    plt.bar(x + 2.5*width, null_val_vals,   width, label="Null Val",   color=null_colors[1], alpha=0.7)
                    plt.bar(x + 3.5*width, null_test_vals,  width, label="Null Test",  color=null_colors[2], alpha=0.7)

                    plt.xticks(x + width, metrics_names, rotation=15)
                    plt.ylabel("Score")
                    plt.title("Classification Metrics by Split: Real vs. Null")
                    plt.legend(fontsize=10, ncol=2)
                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    plt.tight_layout()
                    #plt.show()
                    plt.savefig(os.path.join(out_dir, "metrics_bar.png"))
                    plt.close()


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