import argparse
import sys
import importlib
import numpy as np
import time
from math import sqrt

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Import OLS helpers directly 
from fibonacci_ols import roll_forecast, make_xy

def import_model_func(module_name, func_name):
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

# Dynamically import model functions from files with underscores in their names
cnn_predict = import_model_func('fibonacci_cnn', 'predict_next_numbers')
rnn_predict = import_model_func('fibonacci_rnn', 'predict_next_number')

def run_ols(seq, n_preds):
    from sklearn.linear_model import LinearRegression
    X, y = make_xy(seq)
    ols = LinearRegression().fit(X, y)
    return roll_forecast(ols, seq.copy(), n_preds)

def run_cnn(seq, n_preds):
    return cnn_predict(seq, n_preds)

def run_rnn(seq, n_preds):
    s = list(seq)
    for _ in range(n_preds):
        next_val = rnn_predict(s)
        s.append(round(next_val))
    return s

def fib_truth(seed, n_more):
    t = seed[:]
    for _ in range(n_more):
        t.append(t[-1] + t[-2])
    return t

MODEL_FUNCS = {
    'ols': run_ols,
    'cnn': run_cnn,
    'rnn': run_rnn,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Fibonacci predictors (OLS, CNN, RNN)")
    parser.add_argument('--models', nargs='+', choices=list(MODEL_FUNCS.keys()) + ['all'], default=['all'],
                        help="Which models to run (default: all)")
    parser.add_argument('--n_preds', type=int, default=5, help="How many future numbers to predict (default: 5)")
    parser.add_argument('--plot', action='store_true', help="Plot the results (requires matplotlib)")
    args = parser.parse_args()

    initial_seq = [1, 1, 2, 3, 5, 8, 13, 21]
    models_to_run = list(MODEL_FUNCS.keys()) if 'all' in args.models else args.models

    results = {}
    timings = {}

    for model in models_to_run:
        print(f"\nRunning {model.upper()} model...")
        try:
            start = time.time()
            seq = MODEL_FUNCS[model](initial_seq, args.n_preds)
            timings[model] = time.time() - start
            print(f"{model.upper()} sequence: {seq}")
            results[model] = seq
        except Exception as e:
            print(f"Error running {model}: {e}")

    # ---- Metrics + Plots ----
    if args.plot and HAS_MPL and results:
        # 1) Line plot of predictions
        plt.figure()
        for model, seq in results.items():
            plt.plot(range(1, len(seq)+1), seq, marker='o', label=model.upper())
        plt.title("Fibonacci Predictions by Model")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        # 2) Compute metrics on the predicted tail only
        truth_full = fib_truth(initial_seq, args.n_preds)
        truth_tail = truth_full[-args.n_preds:]  # ground-truth future values

        metrics = {} 
        for model, seq in results.items():
            pred_tail = seq[-args.n_preds:]
            errs = np.array(pred_tail) - np.array(truth_tail)
            mae  = np.mean(np.abs(errs))
            mse  = np.mean(errs**2)
            rmse = sqrt(mse)
            mape = np.mean(np.abs(errs) / np.array(truth_tail)) * 100
            metrics[model] = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE_%": mape,
                "Time_s": timings.get(model, np.nan)
            }

        # 3) Bar chart for a couple of key metrics
        plt.figure()
        models = list(metrics.keys())
        mae_vals  = [metrics[m]["MAE"]  for m in models]
        rmse_vals = [metrics[m]["RMSE"] for m in models]

        x = np.arange(len(models))
        width = 0.35
        plt.bar(x - width/2, mae_vals,  width, label="MAE")
        plt.bar(x + width/2, rmse_vals, width, label="RMSE")
        plt.xticks(x, [m.upper() for m in models])
        plt.ylabel("Error")
        plt.title("Prediction Error Metrics")
        plt.legend()
        plt.tight_layout()

        # Optional: print a small table to stdout
        print("\n=== Metrics ===")
        for m, d in metrics.items():
            print(f"{m.upper():>4}  MAE={d['MAE']:.3f}  RMSE={d['RMSE']:.3f}  "
                  f"MAPE%={d['MAPE_%']:.2f}  Time={d['Time_s']:.4f}s")

        plt.show()

    elif args.plot and not HAS_MPL:
        print("matplotlib is not installed. Install it to enable plotting.")