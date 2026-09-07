"""Offline-Modellvergleich der trainierten CaDeLaC-Varianten (joint vs. base_pos_z).

Bewertet jedes Modell auf Modellebene, d.h. mit einem reinen Forward-Pass ueber den
Testsatz und Ground-Truth-Features (offline). Das isoliert die Modellguete von der
Filterintegration; der Online-Effekt (Feature-Rueckkopplung) wird separat auf
Estimator-Ebene gemessen.

Drei Metriken, passend zu den drei Modell-Level-Vergleichen im Report:

  tau_mse   normierter MSE der Residualdrehmoment-Vorhersage tau~ (= Trainingsziel,
            Gl. eq:loss, auf dem Testsatz). Referenz: Nominalmodell = Residuum 0.
  bias_mse  normierter MSE des Bias-Residuums b~ gegen diff_tau_c + diff_tau_g.
            Vorhersage = Forward-Pass bei qdd = 0 (Euler-Lagrange: tau(qdd=0) = Bias).
  mass_mae  mittlerer |Fehler| der Zusatzmasse M~[0,0] pro Testlauf (Laufmittel gegen
            geloggtes base_mass). Referenz: physikalische Fensterschranke, pro Lauf
            genauso gemittelt (like-for-like).

Ergebnisse werden pro Modell in einer JSON-Datei gecacht.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

BASE_MASS_NOM = 13.042  # aliengo Rumpf ohne Payload (per Gravitations-Fit bestaetigt)


def _felan_imports():
    """felan importieren; dill wird nur fuer .pkl-Datensaetze gebraucht -> stubben."""
    try:
        import dill  # noqa: F401
    except ImportError:
        sys.modules["dill"] = types.SimpleNamespace(load=None, dump=None)
    from felan.train import load_model_fn
    from felan.data_scripts.data_loaders import load_custom_dataset
    from felan.models.np_math_utils import get_euler_from_mj_quat
    from felan.models.log_chol_cadelac_pot_param import CaDeLaCLogChol, get_config_from_dict
    return load_model_fn, load_custom_dataset, get_euler_from_mj_quat, CaDeLaCLogChol, get_config_from_dict


def _build_history(input_values, hists):
    hjp, hjv, hdt, hbo, hbv, hbav, hbz = hists
    # Reihenfolge MUSS dem match-Block in train_quad_cadelac.py entsprechen
    if input_values == "joint":
        return np.concatenate([hjp, hjv, hdt], axis=-1)
    if input_values == "base_pos_z":
        return np.concatenate([hbo, hbv, hbav, hbz, hdt], axis=-1)
    raise ValueError(input_values)


def offline_eval(model_name, model_dir, dataset_path, input_values,
                 base_mass_nom=BASE_MASS_NOM):
    """Ein Modell offline auf dem Testsatz bewerten. Gibt ein Metrik-Dict zurueck."""
    import jax
    import jax.numpy as jnp
    (load_model_fn, load_custom_dataset, get_euler_from_mj_quat,
     CaDeLaCLogChol, get_config_from_dict) = _felan_imports()

    params, hyper = load_model_fn(model_name, str(model_dir))
    tw = hyper["time_window"]
    st, gp = hyper.get("history_stride", 1), hyper.get("history_gap", 0)

    # seed=0: der Split haengt am -s-Argument (Default 0), nicht am Sweep-Seed --
    # alle Modelle teilen denselben Split (train_quad_cadelac.py Z. 70/155/332).
    train_data, test_data, divider, _, test_base_mass = load_custom_dataset(
        str(dataset_path), hist_length=tw, sample_offset=0, seed=0,
        hist_stride=st, hist_gap=gp)
    if list(test_data[0]) != list(hyper["test_labels"]):
        raise RuntimeError(f"Split mismatch fuer {model_name}")

    (tl, qp, qv, qa, tau, tm, tc, tg, *hists) = test_data
    qp = np.hstack((qp[:, :3], get_euler_from_mj_quat(qp[:, 3:7]), qp[:, 7:]))
    hist = _build_history(input_values, hists)

    # Normierung wie im Training: Varianz des TRAIN-Residuums + 1e-2
    norm_tau = np.var(np.asarray(train_data[4]), axis=0) + 1e-2

    model = CaDeLaCLogChol(hyper["nv_dof"], get_config_from_dict(hyper))
    f = jax.jit(model.apply)
    to32 = lambda x: jnp.asarray(np.asarray(x), jnp.float32)
    q_j, qv_j, qa_j, h_j = to32(qp), to32(qv), to32(qa), to32(hist)

    tau_hat, _, extras = f(params, q_j, qv_j, qa_j, h_j)          # volles tau~
    bias_hat, _, _ = f(params, q_j, qv_j, to32(np.zeros_like(np.asarray(qa))), h_j)
    tau_hat, bias_hat = np.asarray(tau_hat), np.asarray(bias_hat)
    M = np.asarray(extras["M"])

    tau_gt = np.asarray(tau)
    bias_gt = np.asarray(tc) + np.asarray(tg)

    tau_mse = float(np.mean(np.sum((tau_hat - tau_gt) ** 2 / norm_tau, axis=1)))
    bias_mse = float(np.mean(np.sum((bias_hat - bias_gt) ** 2 / norm_tau, axis=1)))

    # Masse pro Lauf: Mittel der Zeitreihe M~[0,0] gegen geloggtes Payload
    div = np.asarray(divider)
    pm = M[:, 0, 0]
    mass_err = np.array([pm[div[i]:div[i + 1]].mean() - (test_base_mass[i] - base_mass_nom)
                         for i in range(len(tl))])

    return {
        "model": model_name, "input_values": input_values,
        "tau_mse": tau_mse, "bias_mse": bias_mse,
        "mass_mae": float(np.abs(mass_err).mean()),
        "mass_rmse": float(np.sqrt((mass_err ** 2).mean())),
        "mass_err_per_run": mass_err.tolist(),
        "test_labels": list(tl),
        "tau_rmse_per_dof": np.sqrt(((tau_hat - tau_gt) ** 2).mean(axis=0)).tolist(),
        "bias_rmse_per_dof": np.sqrt(((bias_hat - bias_gt) ** 2).mean(axis=0)).tolist(),
        "tau_rmse_nominal_per_dof": np.sqrt((tau_gt ** 2).mean(axis=0)).tolist(),
    }


def dataset_references(dataset_path, base_mass_nom=BASE_MASS_NOM,
                       hist_length=25, hist_stride=4, hist_gap=10):
    """Modellunabhaengige Referenzwerte: Nominalmodell-MSEs und Fensterschranke."""
    _, load_custom_dataset, _, _, _ = _felan_imports()
    train_data, test_data, divider, _, test_base_mass = load_custom_dataset(
        str(dataset_path), hist_length=hist_length, sample_offset=0, seed=0,
        hist_stride=hist_stride, hist_gap=hist_gap)
    (tl, _, _, _, tau, _, tc, tg, *_) = test_data
    norm_tau = np.var(np.asarray(train_data[4]), axis=0) + 1e-2
    tau_gt, bias_gt = np.asarray(tau), np.asarray(tc) + np.asarray(tg)
    div = np.asarray(divider)
    # Schranke like-for-like: Laufmittel von tau~_z / g gegen wahres Payload
    bound = np.array([tau_gt[div[i]:div[i + 1], 2].mean() / 9.81
                      - (test_base_mass[i] - base_mass_nom) for i in range(len(tl))])
    return {
        "tau_mse_nominal": float(np.mean(np.sum(tau_gt ** 2 / norm_tau, axis=1))),
        "bias_mse_nominal": float(np.mean(np.sum(bias_gt ** 2 / norm_tau, axis=1))),
        "mass_bound_mae": float(np.abs(bound).mean()),
        "mass_bound_rmse": float(np.sqrt((bound ** 2).mean())),
    }


def compare_input_variants(dataset_path, model_dir, name_template, seeds=range(5),
                           variants=("joint", "base_pos_z"), cache_file=None):
    """Alle Modelle bewerten (mit JSON-Cache). Gibt (rows, refs) zurueck."""
    cache = {}
    if cache_file is not None:
        cache_file = Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())
    rows = []
    for variant in variants:
        for seed in seeds:
            name = name_template.format(input_values=variant, seed=seed)
            if name not in cache:
                print(f"  bewerte {variant} seed {seed} ...", flush=True)
                cache[name] = offline_eval(name, model_dir, dataset_path, variant)
                if cache_file is not None:
                    Path(cache_file).write_text(json.dumps(cache))
            rows.append(cache[name])
    refs = dataset_references(dataset_path)
    return rows, refs


def print_comparison(rows, refs):
    """Tabelle pro Modell + Aggregat pro Variante + Referenzen. Gibt bestes Modell zurueck."""
    print(f"\n{'Variante':<12}{'Seed':>5}{'tau-MSE':>10}{'Bias-MSE':>10}"
          f"{'Masse-MAE':>11}{'Masse-RMSE':>12}")
    print("-" * 60)
    by_variant = {}
    for r in rows:
        seed = r["model"].rsplit("_", 1)[-1]
        print(f"{r['input_values']:<12}{seed:>5}{r['tau_mse']:>10.3f}{r['bias_mse']:>10.3f}"
              f"{r['mass_mae']:>11.3f}{r['mass_rmse']:>12.3f}")
        by_variant.setdefault(r["input_values"], []).append(r)
    print("-" * 60)
    for v, rs in by_variant.items():
        for key, label in [("tau_mse", "tau-MSE"), ("bias_mse", "Bias-MSE"),
                           ("mass_mae", "Masse-MAE")]:
            vals = np.array([r[key] for r in rs])
            print(f"{v:<12} {label:<10} {vals.mean():7.3f} +- {vals.std():.3f}   "
                  f"[{vals.min():.3f}, {vals.max():.3f}]")
        print("-" * 60)
    print(f"Referenz Nominalmodell : tau-MSE {refs['tau_mse_nominal']:.3f}   "
          f"Bias-MSE {refs['bias_mse_nominal']:.3f}")
    print(f"Fensterschranke (500ms): Masse-MAE {refs['mass_bound_mae']:.3f} kg "
          f"(RMSE {refs['mass_bound_rmse']:.3f})")

    best = min(rows, key=lambda r: r["mass_mae"])
    print(f"\nBestes Modell (kleinste Masse-MAE): {best['model']}")
    return best

# ---------------------------------------------------------------------------
# Online-Bewertung: Modell laeuft IM Filter (online-realistisch, keine GT-Kanaele).
# Der Unterschied zu offline_eval() ist die Herkunft der Eingaben: hier baut
# run_state_estimation das History-Fenster aus den Groessen, die der laufende
# Filter liefert. Fuer base_pos_z sind das u.a. base_vel und p_z aus carry["x"],
# also Filterschaetzungen -> der LSTM-Kontext ist kontaminiert. Fuer joint sind
# die History-Kanaele direkt gemessen.
# ---------------------------------------------------------------------------

def online_eval(model_name, model_dir, dataset_path, sim_nums):
    """Ein Modell im Filter bewerten. Gibt Massenfehler und KF-Metriken zurueck."""
    from utils.evaluate import evaluate_dataset
    res = evaluate_dataset(dataset_path, Path(model_dir) / model_name,
                           sim_nums=sim_nums, verbose=False)
    with_model = [r for r in res["runs"] if r["has_model"]]
    mass_err = np.array([r["inertia"]["dm_err"] for r in with_model])
    return {
        "model": model_name,
        "mass_mae": float(np.abs(mass_err).mean()),
        "mass_rmse": float(np.sqrt((mass_err ** 2).mean())),
        "mass_err_per_run": mass_err.tolist(),
        "dM_drift": float(np.mean([r["inertia"]["dM_drift"] for r in with_model])),
        "vel_rmse": np.mean([r["kf"]["vel_rmse"] for r in res["runs"]], axis=0).tolist(),
        "sims": [r["sim"] for r in res["runs"]],
    }


def compare_online(dataset_path, model_dir, name_template, sim_nums, seeds=range(5),
                   variants=("joint", "base_pos_z"), cache_file=None):
    """online_eval fuer alle Modelle, mit JSON-Cache. Gibt die Ergebniszeilen zurueck."""
    cache = {}
    if cache_file is not None:
        cache_file = Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())
    rows = []
    for variant in variants:
        for seed in seeds:
            name = name_template.format(input_values=variant, seed=seed)
            if name not in cache:
                print(f"  online: {variant} seed {seed} ...", flush=True)
                r = online_eval(name, model_dir, dataset_path, sim_nums)
                r["input_values"] = variant
                cache[name] = r
                if cache_file is not None:
                    cache_file.write_text(json.dumps(cache))
            rows.append(cache[name])
    return rows


def print_online_comparison(rows_online, rows_offline=None):
    """Tabelle pro Modell + Aggregat pro Variante. Gibt (bestes Modell, Aggregat) zurueck."""
    has_off = rows_offline is not None
    off = {r["model"]: r for r in (rows_offline or [])}
    head = f"\n{'Variante':<12}{'Seed':>5}{'Masse-MAE':>11}{'Masse-RMSE':>12}{'dM-Drift':>10}{'KF vel_z':>10}"
    if has_off:
        head += f"{'(offline)':>11}"
    print(head)
    print("-" * (len(head) - 2))
    by_variant = {}
    for r in rows_online:
        seed = r["model"].rsplit("_", 1)[-1]
        line = (f"{r['input_values']:<12}{seed:>5}{r['mass_mae']:>11.3f}{r['mass_rmse']:>12.3f}"
                f"{r['dM_drift']:>10.3f}{r['vel_rmse'][2]:>10.3f}")
        if has_off:
            line += f"{off[r['model']]['mass_mae']:>11.3f}"
        print(line)
        by_variant.setdefault(r["input_values"], []).append(r)

    print("-" * (len(head) - 2))
    agg = {}
    for v, rs in by_variant.items():
        mae = np.array([r["mass_mae"] for r in rs])
        agg[v] = {"mean": float(mae.mean()), "std": float(mae.std()),
                  "min": float(mae.min()), "max": float(mae.max())}
        extra = ""
        if has_off:
            o = np.array([off[r["model"]]["mass_mae"] for r in rs])
            extra = f"   |  offline {o.mean():.3f} +- {o.std():.3f}"
        print(f"{v:<12} Masse-MAE online  {mae.mean():.3f} +- {mae.std():.3f}   "
              f"[{mae.min():.3f}, {mae.max():.3f}]{extra}")

    best_variant = min(agg, key=lambda v: agg[v]["mean"])
    best = min((r for r in rows_online if r["input_values"] == best_variant),
               key=lambda r: r["mass_mae"])
    print(f"\nBeste Variante (kleinste Online-MAE im Mittel): {best_variant}")
    print(f"Bester Seed darin: {best['model'].rsplit('_', 1)[-1]}  ({best['mass_mae']:.3f} kg)")
    return best, agg
