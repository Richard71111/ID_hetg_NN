import os
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from typing import List, Optional
import train.MLP_TrainTest as MLP_fn
import torch.nn as nn
import torch
import numpy as np
import math

class Single_mesh_DS(torch.utils.data.Dataset):
    def __init__(self, input, target):
        self.input = torch.tensor(input, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

def make_mesh_ds(mesh_idx: int,
                 GJ_coupling:str,
                 normalization: Optional[str] = 'zscore',
                 scaler = None,
                 full_seq = True
                 ):
    if mesh_idx <1 and mesh_idx > 384:
        raise ValueError("Mesh index should in the range of [1,384].")
    
    if not full_seq:
        if GJ_coupling == 'strong':
            df = pd.read_csv(os.path.join("dataset","Strong_GJ_Coupling", f'Mesh_idx_{mesh_idx}.csv'))
        elif GJ_coupling == 'weak':
            df = pd.read_csv(os.path.join("dataset","Weak_GJ_Coupling", f'Mesh_idx_{mesh_idx}.csv'))
    else:
        if GJ_coupling == 'strong':
            df = pd.read_csv(os.path.join("dataset","Strong_GJ_Coupling_full_seq", f'Mesh_idx_{mesh_idx}_full_seq.csv'))
        elif GJ_coupling == 'weak':
            df = pd.read_csv(os.path.join("dataset","Weak_GJ_Coupling_full_seq", f'Mesh_idx_{mesh_idx}_full_seq.csv'))

    if scaler is None:
        print("Scaler is None use new scaler to fit data")
        if normalization == 'zscore':
            scaler = StandardScaler()
        elif normalization == 'minmax':
            scaler = MinMaxScaler()
        elif normalization == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {normalization}")
        scaler.fit(np.array(df.values))
    if normalization is not None:
        data = df.values
        data = scaler.transform(np.array(data[:,:-2]))
    else:
        data = np.array(df.values)
    X = data[:-1, 2:]
    Y = data[1:,  :2]

    ds = Single_mesh_DS(X,Y)

    T = np.array(df['t_save'].values[1:])

    return ds, data, scaler, T

def plot_ID_current(mesh_idx: List[int],
                    model,
                    loss_fn = nn.MSELoss(),
                    scaler = None,
                    normalization: Optional[str] = 'zscore',
                    GJ_coupling = 'strong',
                    full_seq = True
                    ):
    if not mesh_idx:
        raise ValueError("Mesh index list is empty")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_plots = len(mesh_idx)
    n_cols = int(math.ceil(math.sqrt(n_plots)))
    n_rows = int(math.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes = axes.reshape(-1)    

    for i, idx in enumerate(mesh_idx):
        ax = axes[i]

        ds, data, scaler, T = make_mesh_ds(
            idx,
            GJ_coupling,
            normalization=normalization,
            scaler=scaler,
            full_seq = full_seq
        )

        model.eval()
        avg_loss, results = MLP_fn.test_MLP_autoregression(
            model,
            ds,
            device,
            loss_fn
        )
        if normalization is not None:
            data = scaler.inverse_transform(data)
            pre_true  = data[1:, 0]
            post_true = data[1:, 1]
            pre_pred  = np.array(results[0])
            post_pred = np.array(results[1])
            if isinstance(scaler,StandardScaler):
                mean_ = scaler.mean_
                scale_ = scaler.scale_
                pre_mean, post_mean = mean_[0], mean_[1] # type: ignore
                pre_std,  post_std  = scale_[0], scale_[1]# type: ignore

                pre_pred  = np.array(results[0]) * pre_std  + pre_mean
                post_pred = np.array(results[1]) * post_std + post_mean
        else:
            pre_true  = data[1:, 0]
            post_true = data[1:, 1]
            pre_pred  = np.array(results[0])
            post_pred = np.array(results[1])

        ax.plot(T, pre_true,  label='$I_{pre}$ true',        linewidth=1.2)
        ax.plot(T, pre_pred,  '--', label='$I_{pre}$ pred',   linewidth=1.0)
        ax.plot(T, post_true, label='$I_{post}$ true',        linewidth=1.2)
        ax.plot(T, post_pred, '--', label='$I_{post}$ pred',  linewidth=1.0)

        ax.legend(fontsize=8)
        ax.set_title(f'Mesh {idx}  |  MSE fit={avg_loss:.4e}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Current')

    if GJ_coupling == 'strong':
        Big_title = "Strong GJ coupling"
    elif GJ_coupling == 'weak':
        Big_title = "Reduced GJ coupling"
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(Big_title, fontsize=16)
    plt.tight_layout()
    plt.show()