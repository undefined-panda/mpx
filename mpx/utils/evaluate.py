"""Evaluate state estimation with CaDeLaC model across whole dataset

Run run_state_estimation() on each simulation run and measure the following:

1. How good is the model prediction for residual torque, inertia matrix and torque biases? 
   Ground truth is stored in dataset.
2. How good is the state estimation of the Kalman filter with this add-on?

Use in notebook:

    from utils.evaluate import evaluate_dataset, print_report
    res = evaluate_dataset(dataset_path, cadelac_path)
    print_report(res)

Use in cmd from mpx/-dir:

    python -m utils.evaluate --dataset ../custom_datasets/quad_mass_dataset_run7.npz \
        --model trained_models/LogChol-CaDeLaC/LogChol-CaDeLaC/<name>
"""

from pathlib import Path

import numpy as np

from utils.kf_utils import load_custom_dataset
from utils.run_estimation import run_state_estimation
from felan.train import load_model_fn

DOF_LABELS = ["f_x", "f_y", "f_z", "tau_roll", "tau_pitch", "tau_yaw"]
AXES = ["x", "y", "z"]

def _rmse(a, b, axis=0):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))

def _rel_rmse(a, b):
    """RMSE geteilt durch den Effektivwert der Referenz.

    Ohne Normierung sind die sechs Basis-Freiheitsgrade nicht vergleichbar: f_z liegt
    bei ~40 N, tau_yaw bei ~0.02 Nm. Der Nenner ist bewusst der RMS und nicht die
    Streuung -- die Residuen sind offsetdominiert (f_z ist im Wesentlichen dm*g), und
    durch die Streuung geteilt ergaeben sich dreistellige, bedeutungslose Werte.
    0.0 ist perfekt, 1.0 heisst "so gross wie das Signal selbst".
    """
    ref = np.asarray(b)
    return _rmse(a, ref) / np.maximum(np.sqrt(np.mean(ref ** 2, axis=0)), 1e-9)


def _nominal_base_mass(robot):
    from gym_quadruped.quadruped_env import QuadrupedEnv
    return float(QuadrupedEnv(robot=robot).mjModel.body_mass[1])


def _model_metrics(result, data, tau_diff, n, base_mass_nom):
    dM = np.asarray(result["inertia_matrix"]["residual"])
    dq = np.asarray(result["qfrc_bias"]["residual"])

    tau_diff_pred = np.asarray(result["tau_diff_pred"]).reshape(n, 6) # model prediction with base_acc estimation of the KF

    # --- Residual ground truth ---
    dq_gt = (data["diff_tau_c_nom"] + data["diff_tau_g_nom"])[:, :6]  # Coriolis + Gravitation
    tau_diff_gt = tau_diff
    dm_true = float(data["base_mass"][0] - base_mass_nom)
    dm_est = dM[:, 0, 0]
    out = {}

    # ---------------- Modell: Traegheitsmatrix ----------------
    # Nur die Masse hat eine Referenz im Datensatz; Schwerpunkt und Traegheitstensor
    # der Nutzlast werden bei der Datenerzeugung nicht mitgeloggt. Fuer die uebrigen
    # Eintraege pruefen wir stattdessen Konstanz: die Nutzlast aendert sich waehrend
    # eines Laufs nicht, jede Zeitvariation in dM ist daher Schaetzrauschen.
    out["inertia"] = {
        "dm_true": dm_true,
        "dm_mean": float(dm_est.mean()),
        "dm_last": float(dm_est[-100:].mean()),
        "dm_std": float(dm_est.std()),
        "dm_err": float(dm_est[-100:].mean() - dm_true),
        "dm_rel": float(abs(dm_est[-100:].mean() - dm_true) / max(abs(dm_true), 1e-9)),
        "dM_mean_abs": np.abs(dM).mean(axis=0),                       # (6,6)
        "dM_drift": float(np.linalg.norm(dM.std(axis=0))),            # Frobenius der Zeitstreuung
        "coupling_mean_abs": float(np.abs(dM[:, :3, 3:]).mean()),     # Schwerpunktversatz
        "angular_mean_abs": float(np.abs(dM[:, 3:, 3:]).mean()),      # Traegheitsaenderung
    }

    # ---------------- Evaluate torque biases ----------------
    out["bias"] = {
        "rmse": _rmse(dq, dq_gt),
        "rel": _rel_rmse(dq, dq_gt),
        "gt_mean_abs": np.abs(dq_gt).mean(axis=0),
        "est_mean_abs": np.abs(dq).mean(axis=0),
    }

    # ---------------- Evaluate tau_diff ----------------
    tau_true_qdd = np.einsum("bij,bj->bi", dM, data["base_acc"]) + dq
    out["tau_diff"] = {
        "rmse": _rmse(tau_diff_pred, tau_diff_gt),
        "rel": _rel_rmse(tau_diff_pred, tau_diff_gt),
        "rmse_true_qdd": _rmse(tau_true_qdd, tau_diff_gt),
        "gt_std": tau_diff_gt.std(axis=0),
        # zum Vergleich: was das Nominalmodell allein liefern wuerde (Residuum = 0)
        "rmse_nominal": _rmse(np.zeros_like(tau_diff_gt), tau_diff_gt),
    }

    return out


def _kf_metrics(result, data):
    pos_est = np.asarray(result["pos_update"])
    vel_est = np.asarray(result["vel_update"])
    ang_est = np.asarray(result["ang_vel_update"])
    legodom = np.asarray(result["leg_odom_vel"])

    # compensate drift between initial pos of dataset and KF (0,0,0)
    pos_gt = data["base_pos"]
    pos_al = pos_est - pos_est[0]
    pos_gt_al = pos_gt - pos_gt[0]

    kf = {
        "pos_rmse": _rmse(pos_al, pos_gt_al),
        "pos_final_drift": np.abs(pos_al[-1] - pos_gt_al[-1]),
        "pos_drift_per_m": float(np.linalg.norm(pos_al[-1] - pos_gt_al[-1])
                                 / max(np.linalg.norm(np.diff(pos_gt_al, axis=0), axis=1).sum(), 1e-9)),
        "vel_rmse": _rmse(vel_est, data["base_vel"]),
        "ang_vel_rmse": _rmse(ang_est, data["base_ang_vel"]),
        "leg_odom_vel_rmse": _rmse(legodom, data["base_vel"]),
        "vel_gt_std": data["base_vel"].std(axis=0),
        "vel_ratio": _rmse(vel_est, data["base_vel"]) / np.maximum(data["base_vel"].std(axis=0), 1e-9),
        "ang_vel_ratio": _rmse(ang_est, data["base_ang_vel"]) / np.maximum(data["base_ang_vel"].std(axis=0), 1e-9),
        "path_length": float(np.linalg.norm(np.diff(pos_gt_al, axis=0), axis=1).sum()),
    }

    # Kontaktkraefte nur dort bewerten, wo der Fuss wirklich Kontakt hat
    cf_est = np.asarray(result["c_force_update"])
    if cf_est.size:
        mask = data["contact_states"].astype(bool)
        cf_gt = data["contact_forces"]
        if mask.any():
            kf["c_force_rmse"] = float(np.sqrt(np.mean((cf_est[mask] - cf_gt[mask]) ** 2)))
            kf["c_force_rmse_z"] = float(np.sqrt(np.mean((cf_est[mask][:, 2] - cf_gt[mask][:, 2]) ** 2)))
            kf["c_force_gt_std_z"] = float(cf_gt[mask][:, 2].std())

    cf_meas = np.asarray(result["c_force_meas"])
    if cf_meas.size:
        mask = data["contact_states"].astype(bool)
        kf["c_force_meas_rmse_z"] = float(
            np.sqrt(np.mean((cf_meas[mask][:, 2] - data["contact_forces"][mask][:, 2]) ** 2)))

    return kf

def evaluate_run(data, cadelac_path, Q, R, est_mode=4,
                 base_pos=None, base_vel=None, base_acc=None,
                 contact_states=None, base_mass_nom=13.042, oracle=False):
    """Evaluate one run.

    If base_pos, base_vel, base_acc, contact_states are None, the KF uses its own estimations.
    """
    tau = (data["tau_m"] + data["tau_c"] + data["tau_g"])[..., :6]
    tau_nom = (data["tau_m_nom"] + data["tau_c_nom"] + data["tau_g_nom"])[..., :6]
    tau_diff = (data['diff_tau_m_nom'] + data['diff_tau_c_nom'] + data['diff_tau_g_nom'])[..., :6]

    # Oracle: wahre Residuen statt Netz -- Obergrenze fuer den Nutzen eines perfekten
    # Modells. dm ist exakt bekannt (base_mass geloggt); Schwerpunkt/Traegheitsoffsets
    # sind nicht geloggt, daher bleibt der Oracle-Traegheitsblock auf dm*I3 beschraenkt.
    oracle_M_res = oracle_qfrc_res = None
    if oracle:
        if cadelac_path is not None:
            raise ValueError("oracle=True verlangt cadelac_path=None (entweder Netz oder Oracle).")
        dm = float(data["base_mass"][0] - base_mass_nom)
        oracle_M_res = np.diag([dm, dm, dm, 0.0, 0.0, 0.0])
        oracle_qfrc_res = (data["diff_tau_c_nom"] + data["diff_tau_g_nom"])[:, :6]

    result = run_state_estimation(
        dt=data["dt"][0],
        base_orient=data["base_orient"], base_ang_vel=data["base_ang_vel"],
        joint_pos=data["joint_pos"], joint_vel=data["joint_vel"],
        joint_acc=data["joint_acc"], joint_torque=data["joint_torque"],
        base_pos=base_pos, base_vel=base_vel, base_acc=base_acc,
        contact_states=contact_states,
        Q=Q, R=R, est_mode=est_mode, cadelac_path=cadelac_path,
        tau=tau, tau_nominal=tau_nom,
        oracle_M_res=oracle_M_res, oracle_qfrc_res=oracle_qfrc_res,
    )

    n = len(data["base_orient"])
    out = {"n_samples": n, "has_model": "inertia_matrix" in result}

    # add evaluation results for model and KF
    if out["has_model"]:
        out.update(_model_metrics(result, data, tau_diff, n, base_mass_nom))
    out["kf"] = _kf_metrics(result, data)
    return out

def evaluate_dataset(dataset_path, cadelac_path=None, sim_nums=None,
                     Q=(1e-4, 1e-4, 1e-4, 1e-3), R=(0.1, 0.1, 1e-4, 0.01), est_mode=4,
                     use_gt_base_pos=False, use_gt_base_vel=False, use_gt_base_acc=False,
                     use_gt_contact_states=False, robot="aliengo", split_seed=0, 
                     train_size=0.75, verbose=True, oracle=False):
    """Evaluate all runs in the dataset

    use_gt_* flags let the KF use ground truth values instead of its own estimations. 
    Set all to False for realistic online configuration.

    With cadelac_path=None the KF uses the nominal model. 
    """
    dataset_path = Path(dataset_path)
    cadelac_path = Path(cadelac_path) if cadelac_path is not None else None
    n_runs = int(np.load(dataset_path)["time"].shape[0])
    if sim_nums is None:
        sim_nums = range(n_runs)

    # test_idx = _default_test_split(n_runs, split_seed, train_size)
    # load test runs from test_labels in hyper
    _, hyper = load_model_fn(cadelac_path.name, cadelac_path.parent)
    test_idx = [label.split("_")[1] for label in hyper["test_labels"]]
    base_mass_nom = _nominal_base_mass(robot)

    runs = []
    for sim in sim_nums:
        data = load_custom_dataset(dataset_path=dataset_path, sim_num=sim)
        r = evaluate_run(
            data, cadelac_path, list(Q), list(R), est_mode=est_mode,
            base_pos=data["base_pos"] if use_gt_base_pos else None,
            base_vel=data["base_vel"] if use_gt_base_vel else None,
            base_acc=data["base_acc"] if use_gt_base_acc else None,
            contact_states=data["contact_states"] if use_gt_contact_states else None,
            base_mass_nom=base_mass_nom, oracle=oracle,
        )
        r["sim"] = sim
        r["split"] = "test" if sim in test_idx else "train"
        runs.append(r)
        if verbose:
            if r["has_model"]:
                i = r["inertia"]
                print(f"  sim {sim:2d} [{r['split']:5s}] dm {i['dm_last']:6.3f} / {i['dm_true']:6.3f} "
                      f"({i['dm_err']:+.3f})   vel RMSE {np.round(r['kf']['vel_rmse'], 3)}", flush=True)
            else:
                print(f"  sim {sim:2d} [{r['split']:5s}] vel RMSE {np.round(r['kf']['vel_rmse'], 3)}", flush=True)

    if cadelac_path is not None:
        model_label = cadelac_path.name
    elif oracle:
        model_label = "Oracle (wahre Residuen)"
    else:
        model_label = "Nominalmodell (ohne CaDeLaC)"
    return {"runs": runs,
            "model": model_label,
            "dataset": dataset_path.name,
            "base_mass_nom": base_mass_nom, "config": {
                "est_mode": est_mode, "gt_base_pos": use_gt_base_pos, "gt_base_vel": use_gt_base_vel,
                "gt_base_acc": use_gt_base_acc, "gt_contact_states": use_gt_contact_states}}


# ----------------------------------------------------------------------------- Ausgabe

def _fmt(agg, fmt="{:.3f}"):
    """Aggregat als 'train x  test y' -- ueberspringt fehlende Splits."""
    return "  ".join(f"{k} " + fmt.format(v) for k, v in agg.items() if k != "all")


def _agg(runs, getter):
    """Mittelt eine Kennzahl getrennt nach train / test / alle."""
    out = {}
    for split in ("train", "test", "all"):
        sel = [r for r in runs if split == "all" or r["split"] == split]
        if sel:
            out[split] = np.mean([getter(r) for r in sel], axis=0)
    return out


def print_report(res):
    runs = res["runs"]
    cfg = res["config"]
    print("\n" + "=" * 78)
    print(f"Modell    : {res['model']}")
    print(f"Datensatz : {res['dataset']}   ({len(runs)} Laeufe, est_mode {cfg['est_mode']})")
    gt = [k[3:] for k, v in cfg.items() if k.startswith("gt_") and v]
    print(f"Ground Truth an den Filter: {', '.join(gt) if gt else 'keine (online-realistisch)'}")
    print("=" * 78)

    if not runs[0]["has_model"]:
        print("\n(kein CaDeLaC-Modell -- nur Filtermetriken)")
        _print_kf_section(runs)
        return

    # ---------------- 1. Traegheitsmatrix ----------------
    print("\n1) TRAEGHEITSMATRIX -- Zusatzmasse dM[0,0]\n")
    print(f"{'sim':>4} {'split':>6} {'wahr':>8} {'geschaetzt':>11} {'Fehler':>8} {'rel':>7} "
          f"{'Streuung':>9} {'dM-Drift':>9}")
    for r in runs:
        i = r["inertia"]
        print(f"{r['sim']:4d} {r['split']:>6} {i['dm_true']:8.3f} {i['dm_last']:11.3f} "
              f"{i['dm_err']:+8.3f} {i['dm_rel']:6.1%} {i['dm_std']:9.3f} {i['dM_drift']:9.3f}")
    mae = _agg(runs, lambda r: abs(r["inertia"]["dm_err"]))
    print("\n  MAE Zusatzmasse [kg]:  " +
          "   ".join(f"{k} {v:.3f}" for k, v in mae.items()))  # inkl. 'all
    print("  (dM-Drift = zeitliche Streuung des gesamten Residuums; die Nutzlast ist")
    print("   konstant, jeder Wert ueber ~0.05 ist Schaetzrauschen)")

    cm = _agg(runs, lambda r: r["inertia"]["coupling_mean_abs"])
    am = _agg(runs, lambda r: r["inertia"]["angular_mean_abs"])
    print("\n  Kopplungsblock dM[:3,3:] (Schwerpunktversatz), Mittel |.|: " + _fmt(cm))
    print("  Winkelblock    dM[3:,3:] (Traegheitsaenderung), Mittel |.|: " + _fmt(am))
    print("  (fuer beide gibt es keine Referenz im Datensatz -- base_inertia_diag,")
    print("   base_iquat und base_ipos werden bei der Datenerzeugung nicht geloggt)")

    # ---------------- 2. Bias-Kraefte ----------------
    print("\n\n2) BIAS-KRAEFTE -- Residuum gegen diff_tau_c_nom + diff_tau_g_nom\n")
    for split in ("train", "test"):
        sel = [r for r in runs if r["split"] == split]
        if not sel:
            continue
        rmse = np.mean([r["bias"]["rmse"] for r in sel], axis=0)
        rel = np.mean([r["bias"]["rel"] for r in sel], axis=0)
        gt = np.mean([r["bias"]["gt_mean_abs"] for r in sel], axis=0)
        print(f"  {split} ({len(sel)} Laeufe)")
        print(f"    {'':13s}" + "".join(f"{l:>11s}" for l in DOF_LABELS))
        print(f"    {'RMSE':13s}" + "".join(f"{v:11.3f}" for v in rmse))
        print(f"    {'rel. RMSE':13s}" + "".join(f"{v:11.3f}" for v in rel))
        print(f"    {'|GT| Mittel':13s}" + "".join(f"{v:11.3f}" for v in gt))

    # ---------------- 3. tau_diff ----------------
    print("\n\n3) TAU_DIFF -- Netzvorhersage gegen tau - tau_nominal\n")
    for split in ("train", "test"):
        sel = [r for r in runs if r["split"] == split]
        if not sel:
            continue
        rmse = np.mean([r["tau_diff"]["rmse"] for r in sel], axis=0)
        rel = np.mean([r["tau_diff"]["rel"] for r in sel], axis=0)
        true_q = np.mean([r["tau_diff"]["rmse_true_qdd"] for r in sel], axis=0)
        base = np.mean([r["tau_diff"]["rmse_nominal"] for r in sel], axis=0)
        print(f"  {split} ({len(sel)} Laeufe)")
        print(f"    {'':17s}" + "".join(f"{l:>11s}" for l in DOF_LABELS))
        print(f"    {'RMSE':17s}" + "".join(f"{v:11.3f}" for v in rmse))
        print(f"    {'rel. RMSE':17s}" + "".join(f"{v:11.3f}" for v in rel))
        print(f"    {'mit wahrer qdd':17s}" + "".join(f"{v:11.3f}" for v in true_q))
        print(f"    {'nur nominal':17s}" + "".join(f"{v:11.3f}" for v in base))
        print(f"    {'Verbesserung':17s}" + "".join(f"{1 - a / max(b, 1e-9):10.0%} "
                                                    for a, b in zip(rmse, base)))

    # ---------------- 4. Kalman-Filter ----------------
    _print_kf_section(runs)


def _print_kf_section(runs):
    print("\n\n4) KALMAN-FILTER -- Zustandsschaetzung gegen Ground Truth\n")
    print(f"{'sim':>4} {'split':>6} | {'vel RMSE [m/s]':>22} | {'ang vel RMSE [rad/s]':>22} | "
          f"{'Pos-Drift':>10} {'je m':>7}")
    for r in runs:
        k = r["kf"]
        print(f"{r['sim']:4d} {r['split']:>6} | " +
              " ".join(f"{v:6.3f}" for v in k["vel_rmse"]) + "    | " +
              " ".join(f"{v:6.3f}" for v in k["ang_vel_rmse"]) + "    | " +
              f"{np.linalg.norm(k['pos_final_drift']):10.3f} {k['pos_drift_per_m']:6.1%}")

    for name, key, unit in (("Lineargeschwindigkeit", "vel_rmse", "m/s"),
                            ("Winkelgeschwindigkeit", "ang_vel_rmse", "rad/s"),
                            ("Leg-Odometrie (roh)", "leg_odom_vel_rmse", "m/s")):
        a = _agg(runs, lambda r, k=key: r["kf"][k])
        print(f"\n  {name} RMSE [{unit}]   " +
              "   ".join(f"{s}: " + " ".join(f"{v:.3f}" for v in a[s])
                         for s in ("train", "test") if s in a))

    # Ein Verhaeltnis ueber 1.0 heisst: die Schaetzung ist schlechter, als konstant
    # den Mittelwert auszugeben. Das trifft hier typischerweise die z-Komponente.
    vr = _agg(runs, lambda r: r["kf"]["vel_ratio"])
    ar = _agg(runs, lambda r: r["kf"]["ang_vel_ratio"])
    print("\n  RMSE / Streuung der Wahrheit  (>1.0 = schlechter als konstant raten)")
    print("    lin. Geschw.  " + "   ".join(f"{s}: " + " ".join(f"{v:5.2f}" for v in vr[s])
                                            for s in ("train", "test") if s in vr))
    print("    Winkelgeschw. " + "   ".join(f"{s}: " + " ".join(f"{v:5.2f}" for v in ar[s])
                                            for s in ("train", "test") if s in ar))

    if "c_force_rmse_z" in runs[0]["kf"]:
        a = _agg(runs, lambda r: r["kf"]["c_force_rmse_z"])
        sd = _agg(runs, lambda r: r["kf"]["c_force_gt_std_z"])
        print(f"\n  Kontaktkraft Fz RMSE [N] (nur Beine in Kontakt): " + _fmt(a, "{:.1f}") +
              f"   (Streuung der wahren Fz: {sd['all']:.1f})")
        if "c_force_meas_rmse_z" in runs[0]["kf"]:
            m = _agg(runs, lambda r: r["kf"]["c_force_meas_rmse_z"])
            print("  davon schon in der Messung (aus Gelenkmomenten):  " + _fmt(m, "{:.1f}"))

    d = _agg(runs, lambda r: r["kf"]["pos_drift_per_m"])
    print("\n  Positionsdrift je zurueckgelegtem Meter: " + _fmt(d, "{:.1%}"))
    print("\n" + "=" * 78 + "\n")


# ============================================================ Q/R-Sweep ======

Q_NAMES = ["Q_pos", "Q_lin_vel", "Q_ang_vel", "Q_c_force"]
R_NAMES = ["R_unused", "R_lin_vel", "R_ang_vel", "R_c_force"]


def sweep_qr(dataset_path, grid, base_Q=(1e-4, 1e-4, 1e-4, 1e-3),
             base_R=(0.1, 0.1, 1e-4, 0.01), cadelac_path=None,
             sim_nums=(0, 4, 8, 12, 16), est_mode=4, robot="aliengo", verbose=True):
    """Kartesisches Produkt ueber Q/R-Eintraege, ausgewertet auf einer Teilmenge der Laeufe.

    `grid` ist ein Dict von Eintragsnamen auf Wertelisten, z.B.

        {"Q_lin_vel": [1e-4, 1e-3, 1e-2], "R_lin_vel": [0.01, 0.1], "R_c_force": [0.01, 1e3]}

    Erlaubte Namen stehen in Q_NAMES und R_NAMES. Nicht genannte Eintraege bleiben auf
    `base_Q`/`base_R`.

    `cadelac_path=None` laeuft das reine Nominalmodell -- das ist fuer die Filterabstimmung
    das richtige Vorgehen, weil die Filtermetriken vom gelernten Residuum kaum abhaengen
    und der Numpy-Pfad ohne JIT-Kompilierung auskommt.

    Hinweis: `R_unused` ist wirklich unbenutzt. Fuer est_mode 2/3/4 liest der Filter
    `R_diag[1:3]` und `R_diag[3]`, weil es keine Positionsmessung gibt.
    """
    from itertools import product

    dataset_path = Path(dataset_path)
    bad = set(grid) - set(Q_NAMES) - set(R_NAMES)
    if bad:
        raise ValueError(f"Unbekannte Gitternamen: {sorted(bad)}. "
                         f"Erlaubt: {Q_NAMES + R_NAMES}")

    keys = list(grid)
    combos = list(product(*(grid[k] for k in keys)))
    trunk = _nominal_base_mass(robot)

    # Datensaetze einmal laden statt pro Kombination
    datas = {s: load_custom_dataset(dataset_path=dataset_path, sim_num=s) for s in sim_nums}

    rows = []
    for ci, combo in enumerate(combos):
        Q, R = list(base_Q), list(base_R)
        for k, v in zip(keys, combo):
            if k in Q_NAMES:
                Q[Q_NAMES.index(k)] = v
            else:
                R[R_NAMES.index(k)] = v

        per_run = []
        for s in sim_nums:
            d = datas[s]
            r = evaluate_run(d, cadelac_path, Q, R, est_mode=est_mode, base_mass_nom=trunk)
            per_run.append(r["kf"])

        row = {"params": dict(zip(keys, combo)), "Q": Q, "R": R}
        for key in ("vel_rmse", "vel_ratio", "ang_vel_rmse", "leg_odom_vel_rmse"):
            row[key] = np.mean([k[key] for k in per_run], axis=0)
        row["pos_drift_per_m"] = float(np.mean([k["pos_drift_per_m"] for k in per_run]))
        if "c_force_rmse_z" in per_run[0]:
            row["c_force_rmse_z"] = float(np.mean([k["c_force_rmse_z"] for k in per_run]))
        # Gesamtnote: mittleres Verhaeltnis RMSE/Streuung ueber die drei Achsen.
        # Unter 1.0 heisst "besser als konstant raten", das ist die Mindestanforderung.
        row["score"] = float(np.mean(row["vel_ratio"]))
        rows.append(row)
        if verbose:
            print(f"  [{ci + 1:2d}/{len(combos)}] " +
                  "  ".join(f"{k}={v:g}" for k, v in row["params"].items()) +
                  f"   ->  vel/std {np.round(row['vel_ratio'], 2)}  Note {row['score']:.2f}",
                  flush=True)

    return {"rows": rows, "keys": keys, "sim_nums": list(sim_nums),
            "base_Q": list(base_Q), "base_R": list(base_R),
            "model": Path(cadelac_path).name if cadelac_path else "nur Nominalmodell",
            "legodom": np.mean([r["leg_odom_vel_rmse"] for r in rows], axis=0)}


def print_qr_report(sweep, top=None):
    rows = sorted(sweep["rows"], key=lambda r: r["score"])
    keys = sweep["keys"]
    print("\n" + "=" * 96)
    print(f"Q/R-SWEEP   Modell: {sweep['model']}   Laeufe: {sweep['sim_nums']}")
    print(f"Basis Q = {sweep['base_Q']}   R = {sweep['base_R']}")
    print(f"Leg-Odometrie (die Messung, die der Filter bekommt): "
          f"{np.round(sweep['legodom'], 3)} m/s")
    print("=" * 96)
    print("\nSortiert nach Note = Mittel von RMSE/Streuung ueber x, y, z.")
    print("Unter 1.0 heisst besser als konstant raten; die z-Spalte ist das eigentliche Problem.\n")

    head = "".join(f"{k:>13s}" for k in keys)
    print(f"{'Rang':>4} {head}  {'vel RMSE x  y  z':>26} {'RMSE/std x  y  z':>22} {'Note':>6} {'Drift':>7}")
    for i, r in enumerate(rows if top is None else rows[:top]):
        vals = "".join(f"{r['params'][k]:13g}" for k in keys)
        print(f"{i + 1:4d} {vals}  " +
              " ".join(f"{v:7.3f}" for v in r["vel_rmse"]) + "   " +
              " ".join(f"{v:6.2f}" for v in r["vel_ratio"]) +
              f" {r['score']:6.2f} {r['pos_drift_per_m']:6.1%}")

    best, worst = rows[0], rows[-1]
    print(f"\n  bester Satz : " + "  ".join(f"{k}={best['params'][k]:g}" for k in keys))
    print(f"                Q = {best['Q']}   R = {best['R']}")
    print(f"                z-Geschwindigkeit {best['vel_rmse'][2]:.3f} m/s "
          f"(Verhaeltnis {best['vel_ratio'][2]:.2f})")
    print(f"  schlechtester: z-Geschwindigkeit {worst['vel_rmse'][2]:.3f} m/s "
          f"(Verhaeltnis {worst['vel_ratio'][2]:.2f})")
    if "c_force_rmse_z" in best:
        print(f"  Kontaktkraft Fz RMSE beim besten Satz: {best['c_force_rmse_z']:.1f} N")
    print("=" * 96 + "\n")
    return rows


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="Checkpoint-Pfad ohne .pkl")
    p.add_argument("--sims", type=int, nargs="+", default=None)
    p.add_argument("--est-mode", type=int, default=4)
    p.add_argument("--robot", default="aliengo")
    p.add_argument("--gt-base-pos", action="store_true")
    p.add_argument("--gt-base-vel", action="store_true")
    p.add_argument("--gt-base-acc", action="store_true")
    p.add_argument("--gt-contact-states", action="store_true")
    a = p.parse_args()
    res = evaluate_dataset(a.dataset, a.model, sim_nums=a.sims, est_mode=a.est_mode,
                           robot=a.robot, use_gt_base_pos=a.gt_base_pos,
                           use_gt_base_vel=a.gt_base_vel, use_gt_base_acc=a.gt_base_acc,
                           use_gt_contact_states=a.gt_contact_states)
    print_report(res)


if __name__ == "__main__":
    _main()
