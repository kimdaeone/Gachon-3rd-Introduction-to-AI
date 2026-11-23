"""Run the fixed batch-one CPU inference benchmark for the frozen WISDM CNNs."""

import argparse
import os
import platform
import time
from pathlib import Path

from final_wisdm import (
    PROTOCOL, WEEK, digest, frozen_cnns, notebook_functions, read_selection, require, save_csv, save_json,
)


def processor_name():
    if platform.system() == "Windows":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", type=Path, default=WEEK)
    parser.add_argument("--runs", type=Path, default=WEEK / "runs")
    parser.add_argument("--output", type=Path, default=WEEK / "final")
    args = parser.parse_args()
    selection, selection_hash = read_selection(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import numpy as np
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    specs = {row["model"]: row for row in selection["cnn_development"]}
    shape = (1, PROTOCOL["window"], 3)
    rng = np.random.default_rng(0)
    inputs = tf.convert_to_tensor(rng.standard_normal(shape).astype(np.float32))
    rows, summaries = [], []
    for name in frozen_cnns(selection):
        tf.keras.backend.clear_session()
        checkpoint = args.runs / f"{name}_fold0_seed0.weights.h5"
        checkpoint_hash = digest(checkpoint)
        require(checkpoint_hash == selection["development_sha256"][checkpoint.name],
                f"{name}: checkpoint differs from freeze")
        with tf.device("/CPU:0"):
            model = notebook_functions(args.notebooks, name, WIN=PROTOCOL["window"],
                                       MODEL_NAME=name)["build_model"]()
            model.load_weights(checkpoint)
            require(model.count_params() == specs[name]["params"], f"{name}: unexpected parameter count")

            @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32)], autograph=False)
            def predict(batch):
                return model(batch, training=False)

            for _ in range(50):
                result = predict(inputs).numpy()
            require(result.shape == (1, 6) and np.isfinite(result).all(), f"{name}: invalid prediction")
            durations = []
            for trial in range(1, 6):
                for call in range(1, 201):
                    started = time.perf_counter_ns()
                    predict(inputs).numpy()
                    elapsed = time.perf_counter_ns() - started
                    durations.append(elapsed)
                    rows.append(dict(model=name, trial=trial, call=call, elapsed_ns=elapsed))
        q1, median, q3 = (np.percentile(durations, [25, 50, 75]) / 1_000_000).tolist()
        summaries.append(dict(model=name, params=model.count_params(), median_ms=median, q1_ms=q1,
                              q3_ms=q3, iqr_ms=q3 - q1, checkpoint=checkpoint.name,
                              checkpoint_sha256=checkpoint_hash))
        print(f"{name}: {median:.3f} ms, Q1–Q3 {q1:.3f}–{q3:.3f} ms", flush=True)

    raw_path = args.output / "cpu_latency_calls.csv"
    save_csv(raw_path, rows)
    save_json(args.output / "cpu_latency.json",
              dict(selection_sha256=selection_hash, implementation_sha256=digest(Path(__file__)),
                   raw_calls_sha256=digest(raw_path), platform=platform.platform(),
                   processor=processor_name(), logical_cpus=os.cpu_count(),
                   onednn_env=os.environ.get("TF_ENABLE_ONEDNN_OPTS", "TensorFlow default"),
                   python=platform.python_version(), tensorflow=tf.__version__, keras=tf.keras.__version__,
                   numpy=np.__version__, intra_op_threads=1, inter_op_threads=1,
                   omp_threads=1, device="CPU:0", input_shape=list(shape), input_dtype="float32",
                   input="Seed 0 fixed standard-normal synthetic tensor shared by all models",
                   warmup_calls=50, trials=5, calls_per_trial=200,
                   timing="Model tf.function and output tensor to NumPy; excludes loading and preprocessing",
                   summary="Median and linear-interpolated quartiles across 1,000 timed calls per model",
                   models=summaries))


if __name__ == "__main__":
    main()
