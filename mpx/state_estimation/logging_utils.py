import matplotlib.pyplot as plt

COLORS = {
    "leg_odom": "tab:gray",
    "nominal": "tab:blue",
    "cadelac": "tab:orange",
    "oracle": "tab:green",
    "gt": "k",
}

def print_section(header, items):
    """Print a header with a bulleted list of info items below it."""
    items = list(items) if items else []

    content_widths = [len(item) + 4 for item in items] if items else [len("  (none)")]
    width = max(len(header) + 4, *content_widths)

    print(f"\n{f' {header} ':═^{width}}")
    if not items:
        print("  (none)")
    else:
        for item in items:
            print(f"  • {item}")
    print("═" * width)

def _print_rmse_table(metrics, title):
    header = f"{'Quantity':<20} {'Unit':<7} {'x':>10} {'y':>10} {'z':>10} {'‖·‖':>10}"

    line = "─" * len(header)
    title = " "+title+" "

    print(f"\n{title:═^{len(header)}}")
    print(header)
    print(f"{'':═^{len(header)}}")

    for key, val in metrics.items():
        print(f"{key:<20} {val["unit"]:<7} "
                f"{val["rmse"][0]:>10.4f} {val["rmse"][1]:>10.4f} {val["rmse"][2]:>10.4f} "
                f"{val["total"]:>10.4f}")
        print(line)

def _print_drift_table(metrics):
    header = f"{'Metric':<25} {'Value':>12} {'Unit':<10}"
    line = "─" * len(header)
    print(f"\n{' Position Drift ':═^{len(header)}}")
    print(header)
    print(line)
    print(f"{'Trajectory length':<25} {metrics["traj_len"]:>12.4f} {'m':<10}")
    print(f"{'Absolute Traj. Error':<25} {metrics["ate"]:>12.4f} {'m':<10}")
    print(f"{'Drift rate':<25} {metrics["drift_rate"]:>12.4f} {'%':<10}")
    print(line)

def _print_cs_table(metrics):
    header = f"{'Leg':<6} {'Precision':>11} {'Recall':>10} {'F1':>8} {'Contact %':>11}"
    line   = "─" * len(header)
    print(f"\n{' Contact State Classification ':═^{len(header)}}")
    print(header)
    print(line)

    for key, val in metrics.items():
        print(f"{key:<6} {val["precision"]:>11.4f} "
              f"{val["recall"]:>10.4f} {val["f1"]:>8.4f} {val["ratio"]:>10.1f}%")
        print(line)

def _plot_estimation(gt, est, time, name, unit, axis_labels=("x", "y", "z"), show=True, save_path=None):
    """Plot estimation vs. ground truth for a multi-axis signal
    """
    _, D = gt.shape
    assert est.shape == gt.shape, f"gt {gt.shape} and est {est.shape} must match."

    fig, axes = plt.subplots(D, 1, figsize=(10, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, gt[:, i],  label="Ground truth", color=COLORS["gt"], lw=1.5)
        ax.plot(time, est[:, i], label="Estimate",     color=COLORS["nominal"], lw=1.2)
        ax.set_ylabel(f"{axis_labels[i]} [{unit}]" if unit else axis_labels[i])
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{name}: estimate vs. ground truth")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
