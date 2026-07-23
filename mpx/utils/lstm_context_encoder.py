import torch.nn as nn
from utils.kf_utils import load_custom_dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

class ContextEncoderLSTM(nn.Module):
    """History-Encoder for sequence of [q, q_dot, tau_tilde]. Returns a latent 
    representation z of the environment
    """
    def __init__(self, input_size, z_dim=10, hidden_size=10, num_layers=5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # order of output tensor: (batch, seq_len, features)
        )
        self.output_layer = nn.Linear(hidden_size, z_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        z = self.output_layer(last_hidden)
        return z

def create_lstm_dataset(dataset_path, n_h):
    data = load_custom_dataset(dataset_path=Path.cwd().parent / dataset_path)
    num_datasets = data["num_datasets"]
    num_datapoints = data["num_datapoints"]

    xs, ys = [], []
    for sim_num in tqdm(range(num_datasets), desc=f"Creating train dataset"):
        residual_torque_full = (data["diff_tau_m_nom"][sim_num] + data["diff_tau_c_nom"][sim_num] + data["diff_tau_g_nom"][sim_num])
        for i in range(num_datapoints - n_h):
            joint_pos = data["joint_pos"][sim_num][i:i+n_h]
            joint_vel = data["joint_vel"][sim_num][i:i+n_h]
            residual_torque = residual_torque_full[i:i+n_h, :6]

            target = residual_torque_full[i+n_h][:6]

            xs.append(np.concatenate([joint_pos, joint_vel, residual_torque], axis=-1))
            ys.append(target)
    
    X = np.array(xs)
    y = np.array(ys)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
