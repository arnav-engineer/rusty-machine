"""
Benchmark Suite: Rusty Machine vs Scikit-learn

Comprehensive performance and accuracy comparison with:
- Warm-up runs to eliminate cold-start bias
- Multiple timed runs with statistical reporting (mean ± std)
- GPU vs CPU prediction benchmarks
- Coefficient agreement analysis
- Sparsity analysis for L1 regularization

Usage:
    python benchmark.py              # Run all benchmarks
    python benchmark.py --quick      # Smaller datasets for quick validation
"""

import argparse
import time
import sys
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import LogisticRegression as SklearnLogistic
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from rustymachine_api.models import LinearRegression as RustyLinear
    from rustymachine_api.models import LogisticRegression as RustyLogistic
except ImportError:
    print("ERROR: rusty_machine not found. Run: maturin develop --release")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
ALPHA = 0.1
WARMUP_RUNS = 2
TIMED_RUNS = 5

FULL_CONFIG = {
    "ridge": {"samples": 1_000_000, "features": 100, "informative": 75},
    "logistic_l2": {"samples": 500_000, "features": 100, "informative": 50,
                    "epochs": 100, "lr": 0.05, "batch_size": 4096, "momentum": 0.0},
    "logistic_l2_momentum": {"samples": 500_000, "features": 100, "informative": 50,
                             "epochs": 100, "lr": 0.05, "batch_size": 4096, "momentum": 0.9},
    "logistic_l1": {"samples": 500_000, "features": 100, "informative": 50,
                    "epochs": 100, "lr": 0.05, "batch_size": 4096, "momentum": 0.0},
}

QUICK_CONFIG = {
    "ridge": {"samples": 100_000, "features": 50, "informative": 40},
    "logistic_l2": {"samples": 100_000, "features": 50, "informative": 25,
                    "epochs": 50, "lr": 0.05, "batch_size": 2048, "momentum": 0.0},
    "logistic_l2_momentum": {"samples": 100_000, "features": 50, "informative": 25,
                             "epochs": 50, "lr": 0.05, "batch_size": 2048, "momentum": 0.9},
    "logistic_l1": {"samples": 100_000, "features": 50, "informative": 25,
                    "epochs": 50, "lr": 0.05, "batch_size": 2048, "momentum": 0.0},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(title: str):
    w = 90
    print("\n" + "=" * w)
    print(f"{title:^{w}}")
    print("=" * w)


def timed(fn, warmup=WARMUP_RUNS, runs=TIMED_RUNS):
    """Run fn with warmup, then time multiple runs. Returns (mean, std, last_result)."""
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return arr.mean(), arr.std(), result


def coef_agreement(rm_coefs, sk_coefs):
    """Compute R² and max absolute error between two coefficient vectors."""
    sk_c = sk_coefs.flatten()
    r2 = r2_score(sk_c, rm_coefs)
    max_err = np.max(np.abs(rm_coefs - sk_c))
    return r2, max_err


def print_row(name, score, mean_t, std_t, metric_name, speedup=None):
    su = f"{speedup:.1f}x" if speedup else "—"
    print(f"  {name:<32} {metric_name:<10} {score:<10.4f} "
          f"{mean_t:>8.4f}s ± {std_t:.4f}s  {su:>8}")


# ---------------------------------------------------------------------------
# Benchmark: Ridge Regression
# ---------------------------------------------------------------------------

def bench_ridge(cfg):
    header(f"RIDGE REGRESSION  (α={ALPHA}, {cfg['samples']:,} × {cfg['features']})")

    X, y = make_regression(
        n_samples=cfg["samples"], n_features=cfg["features"],
        n_informative=cfg["informative"], noise=25.0, random_state=RANDOM_STATE,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    # --- Training ---
    def train_rusty():
        m = RustyLinear(alpha=ALPHA)
        m.fit(X_train, y_train)
        return m

    def train_sklearn():
        m = SklearnRidge(alpha=ALPHA, solver="auto")
        m.fit(X_train, y_train)
        return m

    rm_mean, rm_std, rm_model = timed(train_rusty)
    sk_mean, sk_std, sk_model = timed(train_sklearn)

    rm_score = r2_score(y_test, rm_model.predict(X_test))
    sk_score = r2_score(y_test, sk_model.predict(X_test))
    speedup = sk_mean / rm_mean if rm_mean > 0 else float("inf")

    print(f"\n  {'Model':<32} {'Metric':<10} {'Score':<10} "
          f"{'Train Time':>18}  {'Speedup':>8}")
    print("  " + "-" * 86)
    print_row("Rusty Machine (GPU)", rm_score, rm_mean, rm_std, "R²", speedup)
    print_row("Scikit-learn  (CPU)", sk_score, sk_mean, sk_std, "R²")

    # Coefficient agreement
    r2_c, max_e = coef_agreement(rm_model.coef_, sk_model.coef_)
    print(f"\n  Coefficient agreement: R²={r2_c:.6f}  max|Δ|={max_e:.6f}")

    # --- Prediction ---
    header("RIDGE PREDICTION BENCHMARK")

    def pred_rusty():
        return rm_model.predict(X_test)

    def pred_sklearn():
        return sk_model.predict(X_test)

    p_rm_mean, p_rm_std, _ = timed(pred_rusty)
    p_sk_mean, p_sk_std, _ = timed(pred_sklearn)
    p_speedup = p_sk_mean / p_rm_mean if p_rm_mean > 0 else float("inf")

    print(f"\n  {'Model':<32} {'Predict Time':>18}  {'Speedup':>8}")
    print("  " + "-" * 62)
    print(f"  {'Rusty Machine (GPU)':<32} {p_rm_mean:>8.4f}s ± {p_rm_std:.4f}s  {p_speedup:>7.1f}x")
    print(f"  {'Scikit-learn  (CPU)':<32} {p_sk_mean:>8.4f}s ± {p_sk_std:.4f}s       —")


# ---------------------------------------------------------------------------
# Benchmark: Logistic Regression
# ---------------------------------------------------------------------------

def bench_logistic(cfg, penalty, label, momentum=0.0):
    header(f"LOGISTIC REGRESSION ({penalty.upper()}, α={ALPHA}, "
           f"momentum={momentum}, {cfg['samples']:,} × {cfg['features']})")

    X, y = make_classification(
        n_samples=cfg["samples"], n_features=cfg["features"],
        n_informative=cfg["informative"], n_redundant=10,
        random_state=RANDOM_STATE,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    C = 1.0 / ALPHA if ALPHA > 0 else 1e6

    def train_rusty():
        m = RustyLogistic(
            epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
            penalty=penalty, alpha=ALPHA, momentum=momentum,
            random_state=RANDOM_STATE,
        )
        m.fit(X_train_s, y_train)
        return m

    def train_sklearn():
        m = SklearnLogistic(
            penalty=penalty, C=C, solver="saga",
            max_iter=cfg["epochs"], tol=1e-4, random_state=RANDOM_STATE,
        )
        m.fit(X_train_s, y_train.ravel())
        return m

    rm_mean, rm_std, rm_model = timed(train_rusty)
    sk_mean, sk_std, sk_model = timed(train_sklearn)

    rm_acc = accuracy_score(y_test, rm_model.predict(X_test_s))
    sk_acc = accuracy_score(y_test, sk_model.predict(X_test_s))
    speedup = sk_mean / rm_mean if rm_mean > 0 else float("inf")

    print(f"\n  {'Model':<32} {'Metric':<10} {'Score':<10} "
          f"{'Train Time':>18}  {'Speedup':>8}")
    print("  " + "-" * 86)
    print_row(f"Rusty Machine (GPU)", rm_acc, rm_mean, rm_std, "Accuracy", speedup)
    print_row(f"Scikit-learn  (CPU)", sk_acc, sk_mean, sk_std, "Accuracy")

    # Coefficient agreement
    r2_c, max_e = coef_agreement(rm_model.coef_, sk_model.coef_)
    print(f"\n  Coefficient agreement: R²={r2_c:.6f}  max|Δ|={max_e:.6f}")

    # Sparsity analysis for L1
    if penalty == "l1":
        rm_zeros = np.sum(np.abs(rm_model.coef_) < 1e-6)
        sk_zeros = np.sum(np.abs(sk_model.coef_.flatten()) < 1e-6)
        print(f"  L1 Sparsity: Rusty={rm_zeros}/{len(rm_model.coef_)} zeros, "
              f"Sklearn={sk_zeros}/{len(sk_model.coef_.flatten())} zeros")

    # --- Prediction ---
    header(f"{label} PREDICTION BENCHMARK")

    def pred_rusty():
        return rm_model.predict(X_test_s)

    def pred_sklearn():
        return sk_model.predict(X_test_s)

    p_rm_mean, p_rm_std, _ = timed(pred_rusty)
    p_sk_mean, p_sk_std, _ = timed(pred_sklearn)
    p_speedup = p_sk_mean / p_rm_mean if p_rm_mean > 0 else float("inf")

    print(f"\n  {'Model':<32} {'Predict Time':>18}  {'Speedup':>8}")
    print("  " + "-" * 62)
    print(f"  {'Rusty Machine (GPU)':<32} {p_rm_mean:>8.4f}s ± {p_rm_std:.4f}s  {p_speedup:>7.1f}x")
    print(f"  {'Scikit-learn  (CPU)':<32} {p_sk_mean:>8.4f}s ± {p_sk_std:.4f}s       —")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rusty Machine Benchmark Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller datasets for quick validation")
    args = parser.parse_args()

    cfg = QUICK_CONFIG if args.quick else FULL_CONFIG
    mode = "QUICK" if args.quick else "FULL"

    print(f"\n{'':=<90}")
    print(f"  RUSTY MACHINE BENCHMARK SUITE ({mode} mode)")
    print(f"  Warm-up: {WARMUP_RUNS} runs | Timed: {TIMED_RUNS} runs | α={ALPHA}")
    print(f"{'':=<90}")

    bench_ridge(cfg["ridge"])
    bench_logistic(cfg["logistic_l2"], "l2", "LOGISTIC L2")
    bench_logistic(cfg["logistic_l2_momentum"], "l2", "LOGISTIC L2+MOMENTUM",
                   momentum=cfg["logistic_l2_momentum"]["momentum"])
    bench_logistic(cfg["logistic_l1"], "l1", "LOGISTIC L1")

    print("\n" + "=" * 90)
    print(f"{'BENCHMARK COMPLETE':^90}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()