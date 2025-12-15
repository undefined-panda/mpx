"""
This file creates plots from custom datasets. The plots are for 'mjx_quad.py'.
"""

import numpy as np
from pathlib import Path

dataset_path = Path.cwd() / "custom_datasets"
dataset = np.load(f"{dataset_path}/quad_dataset_run0.npz")

def base_plot(dataset):
    # base height over time

    # base orientation over time
    pass

def contact_plot(dataset):
    # contact state per feet over time

    # vertical force per feet
    pass

def joint_plot(dataset):
    # joint angle over time

    # joint angle vs joint velocity

    # joint torque over time
    pass

def control_plot(dataset):
    # 

    # joint target position vs actual position
    pass

if __name__ == "__main__":
    num_simulations = len(next(iter(dataset.values())))
    
    base_plot(dataset, num_simulations)