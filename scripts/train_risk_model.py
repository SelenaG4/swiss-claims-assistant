"""
Hyperparameter sweep for the claims risk model, tracked end-to-end in MLflow.

Where scripts/generate_synthetic_claims.py trains one HistGradientBoostingClassifier
with default hyperparameters (the fast path to get the app running), this script is
the "how did we actually pick those numbers" record: a small grid of hyperparameter
combinations, each trained and evaluated as its own tracked MLflow run (params, held-
out AUC, a permutation-importance plot, and the model itself), so the choice behind
models/risk_model.joblib is reproducible and inspectable rather than just asserted.

Run:
    python scripts/train_risk_model.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # then open http://localhost:5000

Tracking uses a local SQLite file (mlflow.db) rather than a tracking server --
zero external services required, same "fully demoable with zero external keys"
principle as the rest of this project. (MLflow's older plain-directory file
store is deprecated as of MLflow 3; sqlite is the current recommended local
backend.)
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

# Same fix as app/risk.py -- must be set before numpy/sklearn/joblib import.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.risk import FEATURE_COLUMNS, MODEL_PATH, save_model  # noqa: E402
from scripts.generate_synthetic_claims import generate_dataset  # noqa: E402

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("claims-risk-model")

# Deliberately small: this is a demonstration of the tracking workflow, not a
# production-scale search. 3 x 2 x 2 = 12 runs, each training in well under a
# second on this synthetic 4000-row dataset.
PARAM_GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_iter": [100, 200],
    "max_depth": [None, 5],
}


def _grid():
    keys = list(PARAM_GRID)
    for values in itertools.product(*PARAM_GRID.values()):
        yield dict(zip(keys, values))


def _importance_plot(model, X_test, y_test, out_dir: Path) -> Path:
    """Permutation importance rather than the model's built-in feature
    importances: HistGradientBoostingClassifier doesn't expose the latter, and
    permutation importance has the advantage of being measured directly against
    the metric that matters here (held-out AUC), not an internal split-gain proxy.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
    )
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([FEATURE_COLUMNS[i] for i in order], result.importances_mean[order], color="#1F3864")
    ax.set_xlabel("Permutation importance (drop in held-out AUC)")
    ax.set_title("Feature importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    df = generate_dataset()
    X, y = df[FEATURE_COLUMNS], df["is_high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tmp_dir = ROOT / "mlruns_tmp"
    tmp_dir.mkdir(exist_ok=True)

    best_auc, best_model, best_params = -1.0, None, None

    for params in _grid():
        run_name = f"lr{params['learning_rate']}_iter{params['max_iter']}_depth{params['max_depth']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_param("n_samples", len(df))

            model = HistGradientBoostingClassifier(random_state=42, **params)
            model.fit(X_train, y_train)

            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            mlflow.log_metric("held_out_auc", auc)

            plot_path = _importance_plot(model, X_test, y_test, tmp_dir)
            mlflow.log_artifact(str(plot_path))
            mlflow.sklearn.log_model(model, name="model")

            print(f"{params} -> held-out AUC {auc:.4f}")

            if auc > best_auc:
                best_auc, best_model, best_params = auc, model, params

    print(f"\nBest run: {best_params} -> held-out AUC {best_auc:.4f}")

    # A separate, clearly-tagged run recording which configuration was actually
    # shipped -- so "what's running in production" is answerable from the MLflow
    # UI itself, not just from reading this script's source after the fact.
    with mlflow.start_run(run_name="selected-production-model"):
        mlflow.log_params(best_params)
        mlflow.log_metric("held_out_auc", best_auc)
        mlflow.set_tag("selected_for_production", "true")
        mlflow.sklearn.log_model(best_model, name="model")

    save_model(best_model)
    print(f"Saved selected model to {MODEL_PATH}")

    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
