"""
This file creates plots from custom datasets. The plots are for 'mjx_quad.py'.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from utils.kf_utils import rmse

def base_plot(time, base_pos):
    # trajectory
    x_pos = base_pos[:, 0]
    y_pos = base_pos[:, 1]

    plt.figure()
    plt.plot(x_pos, y_pos)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory")
    plt.grid(True)
    plt.show()

    # base height over time
    z_pos = base_pos[:, 2]

    plt.figure()
    plt.plot(time, z_pos)
    plt.xlabel("Time")
    plt.ylabel("Base Height")
    plt.title("Base Height over Time")
    plt.grid(True)
    plt.show()

def contact_force_mse(gt_contact_force, estimates):
    """Per-leg, per-axis MSE between several contact-force estimates and ground truth.

    Args:
        gt_contact_force (np.ndarray): ground truth contact force, shape (T, 4, 3)
        estimates (dict[str, np.ndarray]): name -> estimate, each shape (T, 4, 3)

    Returns:
        dict[str, np.ndarray]: name -> MSE per leg/axis, shape (4, 3)
    """

    mse = {name: np.mean((gt_contact_force - est) ** 2, axis=0) for name, est in estimates.items()}

    leg_labels = np.arange(1, 5)
    force_labels = ["x", "y", "z"]
    print("leg-axis".ljust(10) + "".join(f"{name:>18s}" for name in estimates))
    for i in range(4):
        for j in range(3):
            row = f"{leg_labels[i]}-{force_labels[j]}".ljust(10)
            row += "".join(f"{mse[name][i, j]:18.4f}" for name in estimates)
            print(row)

    return mse


def plot_contact_force_comparison(gt_contact_force, estimates, num_point=500, colors=None):
    """Plot several contact-force estimates against ground truth, one subplot per leg/axis.

    Args:
        gt_contact_force (np.ndarray): ground truth contact force, shape (T, 4, 3)
        estimates (dict[str, np.ndarray]): name -> estimate, each shape (T, 4, 3)
        num_point (int): number of samples to plot
    """

    if colors is None: colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    _, axes = plt.subplots(4, 3, figsize=(50, 20))

    leg_labels = np.arange(1, 5)
    force_labels = ["x", "y", "z"]
    for i in range(4):
        for j in range(3):
            axes[i, j].plot(np.arange(num_point), gt_contact_force[:num_point, i, j], "--", label="gt", color="black")
            for k, (name, est) in enumerate(estimates.items()):
                axes[i, j].plot(np.arange(num_point), est[:num_point, i, j], label=name, alpha=0.8, color=colors[k % len(colors)])
            axes[i, j].set_ylabel(f"leg {leg_labels[i]} - {force_labels[j]}")
            axes[i, j].legend()

    plt.tight_layout()
    plt.show()

def compare_estimation_plot(time,
                            gt_values,
                            gt_label, 
                            pred_values, 
                            pred_labels, 
                            ylabel, 
                            title, 
                            output_folder=None, 
                            file_name=None,
                            file_types=["png", "pdf"],
                            colors=None, 
                            combined_plot=True, 
                            time_window=None, 
                            font_size=None,
                            show_legend=True,
                            seperate_legend=False):
    """

    Args:
        time (np.array): Time
        gt_values (np.array): Array of ground truth values
        pred_values (list): List of arrays with predicted values
        pred_labels (list): List with corresponding labels
        ylabel (str): Y-label
        title (str): Title
        combined_plot (bool, optional): If True, plot values of each prediction in one plot, else seperated. Defaults to True.
    """

    if font_size: plt.rcParams.update({'font.size': font_size})

    if len(pred_values) != len(pred_labels):
        print(f"prediction arrays and labels have different len: {len(pred_values)} and {len(pred_labels)}")
        return

    pred_shapes = [arr.shape for arr in pred_values]
    gt_shape = gt_values.shape
    shape_match = all(shape == gt_shape for shape in pred_shapes)
    if not shape_match:
        print(f"gt_values and pred_values have different number of dimensions: {gt_shape} and {pred_shapes}")
        return
    
    if colors is None: colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    num_plots = 1 if combined_plot else len(pred_values)
    num_dim = gt_shape[1]
    if time_window is not None:
        if not isinstance(time_window, list):
            print("time_window must be type list")
            return
        if len(time_window) != 2:
            print("time_window must have 2 values")
            return
    fig_size = (16,8) if combined_plot else (20,15)
    _, axes = plt.subplots(num_plots, 1, figsize=fig_size, sharex=combined_plot)
    ax_list = [axes] if combined_plot else list(axes)

    # ground truth: combined_plot only plots it once (num_plots == 1), separate plots once per subplot
    for j, ax in enumerate(ax_list):
        for i in range(num_dim):
            ax.plot(time, gt_values[:, i], "--", label=gt_label, color=colors[j])

    for k, prediction in enumerate(pred_values):
        ax = ax_list[0] if combined_plot else ax_list[k]
        for i in range(num_dim):
            ax.plot(time, prediction[:, i], linestyle="-", label=f"{pred_labels[k]}", alpha=0.5, color=colors[k+1])

    for ax in ax_list:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        if time_window is not None: ax.set_xlim(time_window)
        if show_legend: ax.legend()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        for file_type in file_types:
            plt.savefig(f"{output_folder}/{file_name}.{file_type}", bbox_inches='tight')

    if seperate_legend:
        legend = ax_list[0].legend()
        fig_leg = plt.figure(figsize=(2,2))
        fig_leg.legend(*ax_list[0].get_legend_handles_labels(), loc="center")
        fig_leg.savefig("legend.png", bbox_inches="tight")
        plt.close(fig_leg)

        if not show_legend: legend.remove()
    plt.tight_layout()
    plt.show()

def estimation_plot(gt_value, est_value, label, start, end, title, calc_rmse=True):
    num_dim = gt_value[0].shape[0]

    fig, axes = plt.subplots(num_dim, 1, figsize=(15,20), constrained_layout=True)
    if calc_rmse: errors = rmse(gt_value, est_value)

    for i in range(num_dim):
        axes[i].plot(np.arange(end-start), est_value[start:end, i], label=f"{label} - kf", color="salmon")
        axes[i].plot(np.arange(end-start), gt_value[start:end, i], label=f"{label} - gt", alpha=0.8, color="green")

        if calc_rmse: axes[i].set_title(f"error: {errors[i]:.4f}")

    fig.suptitle(title, fontsize=16)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right")

    plt.show()

def contact_state_plot(gt_value, est_value, start, end, title, legs=("FL", "FR", "RL", "RR")):
    num_legs = gt_value.shape[1]
    fig, axes = plt.subplots(num_legs, 1, figsize=(15, 6), constrained_layout=True, sharex=True)

    t = np.arange(end - start)
    for i in range(num_legs):
        gt_i = gt_value[start:end, i].astype(int)
        est_i = est_value[start:end, i].astype(int)

        # Step-Plots für binäre Signale
        axes[i].step(t, gt_i, where="post", label="gt", color="green", linewidth=2)
        axes[i].step(t, est_i + 0.05, where="post", label="est", color="salmon", linewidth=1.5)  # kleiner Offset

        # Mismatch als roter Hintergrund
        mismatch = gt_i != est_i
        axes[i].fill_between(t, -0.1, 1.15, where=mismatch, color="red", alpha=0.15, step="post")

        # Accuracy statt RMSE
        acc = np.mean(gt_i == est_i)
        fp = np.mean((gt_i == 0) & (est_i == 1))
        fn = np.mean((gt_i == 1) & (est_i == 0))
        axes[i].set_title(f"{legs[i]}  |  acc: {acc:.3f}  FP: {fp:.3f}  FN: {fn:.3f}")

        axes[i].set_ylim(-0.15, 1.25)
        axes[i].set_yticks([0, 1])
        axes[i].set_ylabel("contact")

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title, fontsize=16)
    plt.show()

def force_estimation_plot(gt_value, est_value, label, start, end, title):
    fig, axes = plt.subplots(4, 3, figsize=(50,20), constrained_layout=True)

    leg_labels = np.arange(start=1, stop=5)
    force_labels = ['x', 'y', 'z']
    for i in range(4):
        for j in range(3):
            axes[i, j].plot(np.arange(end-start), est_value[start:end, i, j], label=f"{label} - kf", color="salmon")
            axes[i, j].plot(np.arange(end-start), gt_value[start:end, i, j], label=f"{label} - gt", alpha=0.8, color="green")
            error = rmse(gt_value[:, i, j], est_value[:, i, j])
            axes[i, j].set_title(f"leg {leg_labels[i]} - {force_labels[j]} | rmse: {error:.4f}")
            axes[i, j].set_ylabel(f'leg {leg_labels[i]} - {force_labels[j]}')
            axes[i, j].legend()

    fig.suptitle(title, fontsize=16)
    plt.show()
