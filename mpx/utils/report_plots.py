"""Report-Auswertung: Vergleichsleiter Leg-Odometrie / KF+Nominal / KF+CaDeLaC / KF+Oracle.

Baut die Abbildungen und Tabellen fuer den finalen Report:

  1. Zustands-Zeitreihen eines Laufs, alle Varianten gegen Ground Truth
     (ein Plot pro Zustandsparameter: lin. Geschwindigkeit, Winkelgeschwindigkeit,
     Position, Kontaktkraefte Fz)
  2. Leiter-Balkendiagramm + Zusammenfassungstabelle ueber den ganzen Datensatz
  3. Massen-Streudiagramm wahr vs. geschaetzt mit physikalischer Schranke
  4. tau_diff-Verbesserung gegenueber dem Nominalmodell pro Kanal

Benutzung im Notebook (siehe kf_with_cadelac.ipynb, Abschnitt "Report"):

    from utils.report_plots import (run_ladder_for_sim, plot_state_comparison,
                                    evaluate_ladder, ladder_summary, plot_ladder_bars,
                                    plot_mass_scatter, plot_tau_improvement)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.run_estimation import run_state_estimation
from utils.evaluate import evaluate_dataset

# Reihenfolge = Leiter von "keine Dynamik" bis "perfekte Dynamik"
VARIANTS = ["leg_odom", "nominal", "cadelac", "oracle"]
LABELS = {
    "leg_odom": "Leg odometry (raw)",
    "nominal": "KF + nominal",
    "cadelac": "KF + CaDeLaC",
    "oracle": "KF + oracle (true residuals)",
    "gt": "Ground truth",
}
COLORS = {
    "leg_odom": "tab:gray",
    "nominal": "tab:blue",
    "cadelac": "tab:orange",
    "oracle": "tab:green",
    "gt": "k",
}

DEFAULT_Q = (1e-4, 1e-4, 1e-4, 1e-3)
DEFAULT_R = (0.1, 0.1, 1e-4, 0.01)
TRUNK_MASS_NOM = 13.042  # aliengo Nominalmasse; bei anderem Roboter anpassen


def _save(fig, save_dir, name):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(save_dir / f"{name}.pdf", bbox_inches="tight")


# --------------------------------------------------------------------------
# 1) Ein Lauf: alle Varianten rechnen + Zustands-Zeitreihen plotten
# --------------------------------------------------------------------------

def run_ladder_for_sim(data, cadelac_path, Q=DEFAULT_Q, R=DEFAULT_R, est_mode=4,
                       trunk_mass_nom=TRUNK_MASS_NOM,
                       variants=("nominal", "cadelac", "oracle"), use_imu=False):
    """Alle Filter-Varianten auf EINEM Lauf (data = load_custom_dataset(..., sim_num=i)).

    Gibt {variante: result-dict von run_state_estimation} zurueck. "leg_odom" braucht
    keinen eigenen Lauf -- die rohe Odometrie liegt in jedem result als "leg_odom_vel".
    Online-realistische Konfiguration: kein Ground-Truth-Kanal an den Filter.
    """
    tau = (data["tau_m"] + data["tau_c"] + data["tau_g"])[..., :6]
    tau_nom = (data["tau_m_nom"] + data["tau_c_nom"] + data["tau_g_nom"])[..., :6]
    common = dict(dt=float(data["dt"][0]), base_orient=data["base_orient"],
                  base_ang_vel=data["base_ang_vel"], joint_pos=data["joint_pos"],
                  joint_vel=data["joint_vel"], joint_acc=data["joint_acc"],
                  joint_torque=data["joint_torque"], Q=list(Q), R=list(R),
                  est_mode=est_mode, tau=tau, tau_nominal=tau_nom)
    if use_imu:
        # IMU-Setup: base_acc als Steuereingang (nur est_mode 1-3 nutzen u=base_acc;
        # in est_mode 4 wird base_acc ignoriert, dort rechnet der Filter die
        # Beschleunigung selbst aus Kraeften und Dynamikmodell).
        common["base_acc"] = data["base_acc"]

    out = {}
    if "nominal" in variants:
        out["nominal"] = run_state_estimation(**common, cadelac_path=None)
    if "cadelac" in variants:
        if cadelac_path is None:
            raise ValueError("Variante 'cadelac' braucht cadelac_path.")
        out["cadelac"] = run_state_estimation(**common, cadelac_path=Path(cadelac_path))
    if "oracle" in variants:
        dm = float(data["base_mass"][0] - trunk_mass_nom)
        out["oracle"] = run_state_estimation(
            **common, cadelac_path=None,
            oracle_M_res=np.diag([dm, dm, dm, 0.0, 0.0, 0.0]),
            oracle_qfrc_res=(data["diff_tau_c_nom"] + data["diff_tau_g_nom"])[:, :6])
    return out


def _timeseries_figure(t, gt, series, title, ylabels, name, save_dir):
    """gt: (N, k). series: {variante: (N, k)}. Ein Subplot pro Komponente."""
    k = gt.shape[1]
    fig, axes = plt.subplots(k, 1, figsize=(11, 2.4 * k), sharex=True)
    axes = np.atleast_1d(axes)
    for j in range(k):
        ax = axes[j]
        ax.plot(t, gt[:, j], color=COLORS["gt"], lw=1.4, label=LABELS["gt"])
        for v in VARIANTS:
            if v in series:
                ax.plot(t, series[v][:, j], color=COLORS[v], lw=1.0, alpha=0.85,
                        label=LABELS[v])
        ax.set_ylabel(ylabels[j])
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    _save(fig, save_dir, name)
    return fig


def plot_state_comparison(data, ladder, sim_label="", save_dir=None):
    """Ein Plot pro Zustandsparameter: alle Varianten + Ground Truth.

    Erzeugt vier Figuren: lineare Geschwindigkeit (inkl. roher Leg-Odometrie),
    Winkelgeschwindigkeit, Position (startwert-bereinigt, wie die Driftmetrik)
    und Kontaktkraefte (4 Beine x 3 Komponenten).
    """
    t = np.asarray(data["time"]) - float(data["time"][0])
    figs = {}

    # --- Lineare Geschwindigkeit (der Parameter, den die Dynamik verbessern soll) ---
    series = {v: np.asarray(r["vel_update"]) for v, r in ladder.items()}
    any_result = next(iter(ladder.values()))
    series["leg_odom"] = np.asarray(any_result["leg_odom_vel"])
    figs["vel"] = _timeseries_figure(
        t, np.asarray(data["base_vel"]), series,
        f"Linear velocity {sim_label}",
        ["$v_x$ [m/s]", "$v_y$ [m/s]", "$v_z$ [m/s]"], f"vel{sim_label}", save_dir)

    # --- Winkelgeschwindigkeit ---
    series = {v: np.asarray(r["ang_vel_update"]) for v, r in ladder.items()}
    figs["ang_vel"] = _timeseries_figure(
        t, np.asarray(data["base_ang_vel"]), series,
        f"Angular velocity {sim_label}",
        ["$\\omega_x$ [rad/s]", "$\\omega_y$ [rad/s]", "$\\omega_z$ [rad/s]"],
        f"ang_vel{sim_label}", save_dir)

    # --- Position, startwert-bereinigt (Drift ist die relevante Groesse) ---
    gt_pos = np.asarray(data["base_pos"]); gt_pos = gt_pos - gt_pos[0]
    series = {}
    for v, r in ladder.items():
        p = np.asarray(r["pos_update"]); series[v] = p - p[0]
    figs["pos"] = _timeseries_figure(
        t, gt_pos, series, f"Position {sim_label}",
        ["$x$ [m]", "$y$ [m]", "$z$ [m]"], f"pos{sim_label}", save_dir)

    # --- Kontaktkraefte: 4 Beine x 3 Kraftkomponenten (Layout wie
    # force_estimation_plot in plot_data.py), alle Leiter-Varianten uebereinander.
    # Der RMSE im Subplot-Titel ist ueber den ganzen Lauf gerechnet, nicht nur ueber
    # das gezeigte Fenster -- und pro Variante, damit die Sprossen vergleichbar sind.
    gt_f = np.asarray(data["contact_forces"]).reshape(len(t), 4, 3)   # (N, 4, 3)
    est_f = {}
    for v in VARIANTS:
        if v in ladder:
            cf = np.asarray(ladder[v]["c_force_update"])
            if cf.size:
                est_f[v] = cf.reshape(len(t), 4, 3)

    force_labels = ["x", "y", "z"]
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True, constrained_layout=True)
    for leg in range(4):
        for comp in range(3):
            ax = axes[leg, comp]
            ax.plot(t, gt_f[:, leg, comp], color=COLORS["gt"], lw=1.4, label=LABELS["gt"])
            rmse_txt = []
            for v, arr in est_f.items():
                ax.plot(t, arr[:, leg, comp], color=COLORS[v], lw=1.0, alpha=0.85,
                        label=LABELS[v])
                rmse = float(np.sqrt(np.mean((arr[:, leg, comp] - gt_f[:, leg, comp]) ** 2)))
                rmse_txt.append(f"{LABELS[v].replace('KF + ', '')} {rmse:.2f}")
            # ax.set_title(f"leg {leg + 1} - {force_labels[comp]}"
            #              + (f" | RMSE: {', '.join(rmse_txt)}" if rmse_txt else ""),
            #              fontsize=8)
            ax.set_title(f"leg {leg + 1} - {force_labels[comp]}", fontsize=8)
            ax.set_ylabel(f"leg {leg + 1} - $F_{force_labels[comp]}$ [N]", fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    for comp in range(3):
        axes[-1, comp].set_xlabel("Time [s]")
    fig.suptitle(f"Contact forces {sim_label}", fontsize=13)
    _save(fig, save_dir, f"contact_f{sim_label}")
    figs["contact_f"] = fig

    return figs


# --------------------------------------------------------------------------
# 2) Ganzer Datensatz: Leiter-Metriken, Tabelle, Balkendiagramm
# --------------------------------------------------------------------------

def evaluate_ladder(dataset_path, cadelac_path, sim_nums=None, Q=DEFAULT_Q, R=DEFAULT_R,
                    est_mode=4, variants=("nominal", "cadelac", "oracle"), **kwargs):
    """evaluate_dataset() fuer jede Leiter-Stufe. Achtung: rechnet den Filter
    len(variants) mal ueber alle Laeufe -- fuer schnelle Iteration sim_nums setzen."""
    res = {}
    if "nominal" in variants:
        print("### Stufe: KF + Nominalmodell")
        res["nominal"] = evaluate_dataset(dataset_path, None, sim_nums=sim_nums,
                                          Q=Q, R=R, est_mode=est_mode, **kwargs)
    if "cadelac" in variants:
        print("### Stufe: KF + CaDeLaC")
        res["cadelac"] = evaluate_dataset(dataset_path, cadelac_path, sim_nums=sim_nums,
                                          Q=Q, R=R, est_mode=est_mode, **kwargs)
    if "oracle" in variants:
        print("### Stufe: KF + Oracle (wahre Residuen)")
        res["oracle"] = evaluate_dataset(dataset_path, None, sim_nums=sim_nums,
                                         Q=Q, R=R, est_mode=est_mode, oracle=True, **kwargs)
    return res


def _split_mean(runs, fn, split):
    vals = [fn(r) for r in runs if r["split"] == split]
    return np.mean(vals, axis=0) if vals else None


def ladder_summary(res_by_variant):
    """Zusammenfassungstabelle der Leiter, getrennt Train/Test. Gibt das
    Zahlen-Dict zurueck (fuer eigene Tabellen im Report)."""
    metrics = {
        "vel RMSE x [m/s]": lambda r: r["kf"]["vel_rmse"][0],
        "vel RMSE y [m/s]": lambda r: r["kf"]["vel_rmse"][1],
        "vel RMSE z [m/s]": lambda r: r["kf"]["vel_rmse"][2],
        "ang vel RMSE [rad/s]": lambda r: np.mean(r["kf"]["ang_vel_rmse"]),
        "Pos-Drift [%/m]": lambda r: 100.0 * r["kf"]["pos_drift_per_m"],
        "Fz RMSE [N]": lambda r: r["kf"].get("c_force_rmse_z", np.nan),
    }
    table = {}
    # Leg-Odometrie ist in jeder Variante identisch enthalten -> aus der ersten ziehen
    first = next(iter(res_by_variant.values()))["runs"]
    for split in ("train", "test"):
        lo = _split_mean(first, lambda r: r["kf"]["leg_odom_vel_rmse"], split)
        if lo is not None:
            table[("leg_odom", split)] = {
                "vel RMSE x [m/s]": lo[0], "vel RMSE y [m/s]": lo[1], "vel RMSE z [m/s]": lo[2],
                "ang vel RMSE [rad/s]": np.nan, "Pos-Drift [%/m]": np.nan, "Fz RMSE [N]": np.nan}
        for v, res in res_by_variant.items():
            row = {name: float(np.round(_split_mean(res["runs"], fn, split), 4))
                   for name, fn in metrics.items()
                   if _split_mean(res["runs"], fn, split) is not None}
            if row:
                table[(v, split)] = row

    names = list(metrics.keys())
    print(f"\n{'Variante':<32}{'Split':<7}" + "".join(f"{n.split(' [')[0]:>16}" for n in names))
    for (v, split), row in table.items():
        print(f"{LABELS.get(v, v):<32}{split:<7}" +
              "".join(f"{row.get(n, float('nan')):>16.4f}" for n in names))
    return table


def plot_ladder_bars(res_by_variant, save_dir=None):
    """Leiter-Balkendiagramm: vel-RMSE pro Achse und Pos-Drift, Train/Test."""
    variants = [v for v in VARIANTS[1:] if v in res_by_variant]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    specs = [("vel RMSE x [m/s]", lambda r: r["kf"]["vel_rmse"][0]),
             ("vel RMSE y [m/s]", lambda r: r["kf"]["vel_rmse"][1]),
             ("vel RMSE z [m/s]", lambda r: r["kf"]["vel_rmse"][2]),
             ("Pos-Drift [%/m]", lambda r: 100.0 * r["kf"]["pos_drift_per_m"])]
    width = 0.35
    for ax, (name, fn) in zip(axes, specs):
        for si, split in enumerate(("train", "test")):
            vals = [_split_mean(res_by_variant[v]["runs"], fn, split) for v in variants]
            x = np.arange(len(variants)) + (si - 0.5) * width
            ax.bar(x, vals, width, label=split,
                   color=["#4878a8", "#e88b4e"][si], alpha=0.9)
        # Referenzlinie: rohe Leg-Odometrie (nur fuer vel-Panels definiert)
        if name.startswith("vel"):
            axis = {"x": 0, "y": 1, "z": 2}[name.split(" ")[2]]
            lo = _split_mean(next(iter(res_by_variant.values()))["runs"],
                             lambda r: r["kf"]["leg_odom_vel_rmse"][axis], "test")
            if lo is not None:
                ax.axhline(lo, color="tab:gray", ls="--", lw=1.2,
                           label="Leg-Odom roh (Test)")
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels([LABELS[v].replace("KF + ", "") for v in variants],
                           rotation=15, ha="right", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Vergleichsleiter: Zustandsschaetzung", y=1.03)
    fig.tight_layout()
    _save(fig, save_dir, "ladder_bars")
    return fig


# --------------------------------------------------------------------------
# 3) Massen-Streudiagramm  /  4) tau_diff-Verbesserung pro Kanal
# --------------------------------------------------------------------------

def plot_mass_scatter(res_cadelac, bound_kg=None, save_dir=None):
    """Wahre vs. geschaetzte Zusatzmasse pro Lauf (Train/Test), Identitaetslinie,
    optional die physikalische Fensterschranke als Band um die Identitaet."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = [0.0, 0.1]
    for split, color, marker in (("train", "#4878a8", "o"), ("test", "#e88b4e", "s")):
        runs = [r for r in res_cadelac["runs"] if r["split"] == split and r["has_model"]]
        x = [r["inertia"]["dm_true"] for r in runs]
        y = [r["inertia"]["dm_last"] for r in runs]
        ax.scatter(x, y, c=color, marker=marker, s=45, alpha=0.85, edgecolors="k",
                   linewidths=0.4, label=f"{split} ({len(runs)} Laeufe)")
        lims = [min(lims[0], *x, *y), max(lims[1], *x, *y)]
    pad = 0.05 * (lims[1] - lims[0])
    lo, hi = lims[0] - pad, lims[1] + pad
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, label="Identitaet")
    if bound_kg is not None:
        ax.fill_between([lo, hi], [lo - bound_kg, hi - bound_kg],
                        [lo + bound_kg, hi + bound_kg], color="tab:green", alpha=0.15,
                        label=f"Fensterschranke $\\pm${bound_kg:g} kg")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("wahre Zusatzmasse $\\Delta m$ [kg]")
    ax.set_ylabel("geschaetzte Zusatzmasse $\\hat{\\Delta m}$ [kg]")
    ax.set_title("Massenschaetzung pro Lauf")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_dir, "mass_scatter")
    return fig


def plot_tau_improvement(res_cadelac, save_dir=None):
    """Verbesserung der tau_diff-Vorhersage gegenueber dem Nominalmodell pro Kanal
    (positiv = Netz besser als keine Korrektur), Train/Test nebeneinander."""
    labels = ["$f_x$", "$f_y$", "$f_z$", "$\\tau_{roll}$", "$\\tau_{pitch}$", "$\\tau_{yaw}$"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.35
    for si, split in enumerate(("train", "test")):
        runs = [r for r in res_cadelac["runs"] if r["split"] == split and r["has_model"]]
        rmse = np.mean([r["tau_diff"]["rmse"] for r in runs], axis=0)
        base = np.mean([r["tau_diff"]["rmse_nominal"] for r in runs], axis=0)
        impr = 100.0 * (1.0 - rmse / np.maximum(base, 1e-9))
        x = np.arange(6) + (si - 0.5) * width
        ax.bar(x, impr, width, label=split, color=["#4878a8", "#e88b4e"][si], alpha=0.9)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(np.arange(6)); ax.set_xticklabels(labels)
    ax.set_ylabel("Verbesserung vs. Nominal [%]")
    ax.set_title("$\\tau_{diff}$-Vorhersage: Verbesserung gegenueber Nominalmodell\n"
                 "(negativ = Netzkorrektur schadet in diesem Kanal)", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_dir, "tau_improvement")
    return fig
