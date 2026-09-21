"""dino-wm dataset loader for CartPole.

Copy this file to  <dino_wm_repo>/datasets/cartpole_dset.py

Data layout (produced by experiments/convert_cartpole_to_dinowm.py):
  states.pth             [N, T_max, 4]   float32
  actions.pth            [N, T_max, 1]   float32  (coarse timestep, one per stored step)
  seq_lengths.pth        [N]             int64
  obses/episode_000.pth  [T, H, W, C]   uint8

Training command (from dino_wm root):
  python train.py env=cartpole frameskip=1 num_hist=3 num_pred=1 img_size=128
"""
import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional

from .traj_dset import TrajDataset, get_train_val_sliced


class CartpoleDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/dinowm_cartpole",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        states = torch.load(self.data_path / "states.pth").float()
        self.states = states
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth")

        n = n_rollout if n_rollout else len(self.states)
        self.states      = self.states[:n]
        self.actions     = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios    = self.states.clone()   # use state as proprio
        print(f"Loaded {n} rollouts")

        self.action_dim  = self.actions.shape[-1]   # 1 for CartPole
        self.state_dim   = self.states.shape[-1]    # 4 for CartPole
        self.proprio_dim = self.proprios.shape[-1]  # 4 for CartPole

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(
                self.actions, self.seq_lengths)
            self.state_mean,  self.state_std  = self.get_data_mean_std(
                self.states,  self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(
                self.proprios, self.seq_lengths)
        else:
            self.action_mean  = torch.zeros(self.action_dim)
            self.action_std   = torch.ones(self.action_dim)
            self.state_mean   = torch.zeros(self.state_dim)
            self.state_std    = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std  = torch.ones(self.proprio_dim)

        self.actions  = (self.actions  - self.action_mean)  / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            all_data.append(data[traj, :traj_len])
        all_data = torch.vstack(all_data)
        return torch.mean(all_data, dim=0), torch.std(all_data, dim=0)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames):
        obs_dir = self.data_path / "obses"
        image  = torch.load(obs_dir / f"episode_{idx:03d}.pth")
        proprio = self.proprios[idx, frames]
        act     = self.actions[idx, frames]
        state   = self.states[idx, frames]

        image = image[frames]           # [T, H, W, C] uint8
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_cartpole_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/dinowm_cartpole",
    normalize_action=False,
    split_ratio=0.9,
    num_hist=3,
    num_pred=1,
    frameskip=1,
):
    dset = CartpoleDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    datasets  = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train,   "valid": dset_val}
    return datasets, traj_dset
