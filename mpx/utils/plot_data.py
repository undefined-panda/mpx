"""
This file creates plots from custom datasets. The plots are for 'mjx_quad.py'.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

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

def compare_estimation_plot(time, 
                            gt_values, 
                            gt_label, 
                            pred_values, 
                            pred_labels, 
                            ylabel, 
                            title, 
                            output_folder=None, 
                            file_name=None, 
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

    for j in range(num_plots):
        for i in range(num_dim):
            if combined_plot:
                axes.plot(time, gt_values[:, i], "--", label=gt_label, color=colors[j])
            else:
                axes[j].plot(time, gt_values[:, i], "--", label=gt_label, color=colors[j])

    if combined_plot:
        for k, prediction in enumerate(pred_values):
            for i in range(num_dim):
                axes.plot(time, prediction[:, i], linestyle="-", label=f"{pred_labels[k]}", alpha=0.5, color=colors[k+1])
    else:
        for j in range(num_plots):
            prediction = pred_values[j]
            for i in range(num_dim):
                axes[j].plot(time, prediction[:, i], linestyle="-", label=f"{pred_labels[j]}", alpha=0.5, color=colors[j+1])

    if combined_plot:
        axes.set_xlabel("Time [s]")
        axes.set_ylabel(ylabel)
        axes.set_title(title)
        axes.grid(True)
        if time_window is not None: axes.set_xlim(time_window)
        if show_legend: axes.legend()
    else:
        for j in range(num_plots):
            axes[j].set_xlabel("Time [s]")
            axes[j].set_ylabel(ylabel)
            axes[j].set_title(title)
            axes[j].grid(True)
            if time_window is not None: axes[j].set_xlim(time_window)
            if show_legend: axes[j].legend()
    
    if output_folder is not None: 
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(output_folder+"/"+file_name+".png", bbox_inches='tight')
        plt.savefig(output_folder+"/"+file_name+".pdf", bbox_inches='tight')

    if seperate_legend: 
        legend = axes.legend()
        fig_leg = plt.figure(figsize=(2,2))
        fig_leg.legend(*axes.get_legend_handles_labels(), loc="center")
        fig_leg.savefig("legend.png", bbox_inches="tight")
        plt.close(fig_leg)

        if not show_legend: legend.remove()
    plt.tight_layout()
    plt.show()
