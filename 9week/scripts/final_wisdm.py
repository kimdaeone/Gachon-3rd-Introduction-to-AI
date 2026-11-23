"""Freeze WISDM model selection, refit on development users, and score the holdout."""

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import platform
import re
import statistics
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


WEEK = Path(__file__).resolve().parents[1]
CNN = ("A0", "A1", "A2", "B1", "B2", "B3", "B4")
ABLATION = "B3-C"
SEEDS = (0, 1, 2)
ACTIVITIES = ("Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing")
GROUPS = [[1, 7, 20, 22, 25, 36], [9, 16, 19, 26, 28, 33], [2, 3, 4, 14, 30, 31],
          [8, 10, 11, 12, 13, 15], [5, 17, 18, 23, 32, 35]]
TEST_USERS = [6, 21, 24, 27, 29, 34]
DATA_SHA = "e573791269aab4a629721cf04b6031b7cbf8e14261cd2bc86e65887a2bc592d7"
PROTOCOL = dict(revision="2026-11-16-v1", data_sha256=DATA_SHA, groups=GROUPS,
                test_users=TEST_USERS, window=128, strides=[64, 128],
                max_gap_ns=1_000_000_000, optimizer="Adam", lr=0.001, batch_size=64,
                max_epochs=100, monitor="val_macro_f1", patience=12)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    temporary.replace(path)


def notebook_code(folder, model, index):
    if model == "RF":
        paths = list(folder.glob("03R_*.ipynb"))
    elif model == ABLATION:
        paths = list(folder.glob(f"10_{ABLATION}_*.ipynb"))
    else:
        paths = list(folder.glob(f"{CNN.index(model) + 3:02d}_{model}_*.ipynb"))
    require(len(paths) == 1, f"Expected one source notebook for {model} in {folder}")
    cell = read_json(paths[0])["cells"][index]
    require(cell["cell_type"] == "code", f"Expected code in {paths[0].name} cell {index}")
    return "".join(cell["source"])


def code_hashes(folder, cnn):
    hashes = {f"{model}_model": source_digest(notebook_code(folder, model, 7))
              for model in (*cnn, "RF")}
    hashes.update({f"A0_data_{index}": source_digest(notebook_code(folder, "A0", index))
                   for index in (4, 5)})
    return hashes


def complete_keys(rows, model):
    keys = {(int(row["fold"]), int(row["seed"])) for row in rows if row["model"] == model}
    require(len(rows) == 15 and keys == {(fold, seed) for fold in range(5) for seed in SEEDS},
            f"{model} needs all 15 distinct development fold/seed runs")


def freeze(args):
    hashes = {}

    def development_run(name):
        runs = args.runs / f"{name}_runs.csv"
        with runs.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        complete_keys(rows, name)
        hashes[runs.name] = digest(runs)
        fold_means = [
            statistics.mean(float(row["macro_f1"]) for row in rows if int(row["fold"]) == fold)
            for fold in range(5)
        ]
        require(all(0 <= float(row["macro_f1"]) <= 1 for row in rows), f"{name}: invalid F1")
        info = dict(model=name, fold_macro_f1=fold_means, macro_f1=statistics.mean(fold_means),
                    fold_std=statistics.stdev(fold_means))
        if name != "RF":
            progress = args.runs / f"{name}_progress.json"
            saved = read_json(progress)
            complete_keys(saved["rows"], name)
            require(saved["protocol"].get("model") == name and
                    all(saved["protocol"].get(key) == value for key, value in PROTOCOL.items()),
                    f"{name}: unexpected development protocol")
            by_key = {(row["fold"], row["seed"]): row for row in saved["rows"]}
            for row in rows:
                fold, seed = int(row["fold"]), int(row["seed"])
                expected = by_key[fold, seed]
                require(all(float(row[col]) == expected[col] for col in
                            ("macro_f1", "accuracy", "best_epoch", "epochs", "params")),
                        f"{name} fold {fold} seed {seed}: CSV and progress disagree")
                require(1 <= int(row["best_epoch"]) <= int(row["epochs"]) <= PROTOCOL["max_epochs"],
                        f"{name} fold {fold} seed {seed}: invalid epoch")
            params = {int(row["params"]) for row in rows}
            require(len(params) == 1 and min(params) > 0, f"{name}: inconsistent parameter counts")
            best_epochs = [by_key[fold, seed]["best_epoch"] for fold in range(5) for seed in SEEDS]
            info.update(params=params.pop(), best_epochs=best_epochs,
                        refit_epochs=int(statistics.median(best_epochs) + 0.5))
            hashes[progress.name] = digest(progress)
            checkpoint = args.runs / f"{name}_fold0_seed0.weights.h5"
            hashes[checkpoint.name] = digest(checkpoint)
        return info

    cnn_rows = [development_run(name) for name in CNN]
    by_model = {row["model"]: row for row in cnn_rows}
    delta = [b3 - b2 for b3, b2 in zip(by_model["B3"]["fold_macro_f1"], by_model["B2"]["fold_macro_f1"])]
    ablation_gate = dict(mean_delta=statistics.mean(delta), improved_folds=sum(d > 0 for d in delta))
    ablation_gate["triggered"] = ablation_gate["mean_delta"] >= 0.005 and ablation_gate["improved_folds"] >= 4
    cnn = CNN
    if ablation_gate["triggered"]:
        cnn_rows.append(development_run(ABLATION))
        cnn = (*CNN, ABLATION)
    rf_row = development_run("RF")

    rf_path = args.runs / "RF_reproduction.json"
    rf = read_json(rf_path)
    require(rf["archive_sha256"] == DATA_SHA and rf["groups"] == GROUPS and
            rf["test_users"] == TEST_USERS, "RF development protocol differs")
    rf_params = dict(rf["model_params"])
    rf_params.pop("random_state")
    hashes[rf_path.name] = digest(rf_path)
    cnn_ranking = sorted(cnn_rows, key=lambda row: (-row["macro_f1"], row["params"], row["model"]))
    winner = cnn_ranking[0]["model"]
    final_models = list(dict.fromkeys([winner, "A0", "RF"]))
    result = dict(schema=1, frozen_at=utc_now(), selection=winner,
                  rule="Highest full-precision development macro F1; fewer parameters; model name",
                  epoch_rule="Median of 15 development best epochs, round half up",
                  final_models=final_models, seeds=list(SEEDS), protocol=PROTOCOL, b3_c_gate=ablation_gate,
                  cnn_development=cnn_ranking, rf_development=rf_row, rf_params=rf_params,
                  source_sha256=code_hashes(args.notebooks, cnn), development_sha256=hashes,
                  implementation_sha256=digest(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "selection.json"
    if path.exists():
        saved = read_json(path)
        result["frozen_at"] = saved["frozen_at"]
        require(saved == result, "Selection is already frozen with different inputs")
    else:
        save_json(path, result)
    print(f"Frozen CNN {winner}, final models {', '.join(final_models)}: {path}", flush=True)


def frozen_cnns(selection):
    return (*CNN, ABLATION) if selection["b3_c_gate"]["triggered"] else CNN


def read_selection(args):
    path = args.output / "selection.json"
    selection = read_json(path)
    require(selection["schema"] == 1 and selection["protocol"] == PROTOCOL and
            selection["seeds"] == list(SEEDS), "Unexpected frozen protocol")
    require(selection["source_sha256"] == code_hashes(args.notebooks, frozen_cnns(selection)),
            "Notebook sources changed after freeze")
    require(selection["implementation_sha256"] == digest(Path(__file__)), "Evaluation code changed after freeze")
    return selection, digest(path)


def preparer(args, np):
    require(digest(args.archive) == DATA_SHA, "WISDM archive SHA-256 differs from development")
    context = dict(ARCHIVE=args.archive, np=np, re=re, tarfile=tarfile, ACTIVITIES=ACTIVITIES,
                   MAX_GAP_NS=PROTOCOL["max_gap_ns"], WIN=PROTOCOL["window"],
                   EVAL_STRIDE=PROTOCOL["strides"][1])
    source = notebook_code(args.notebooks, "A0", 4)
    module = ast.parse(source)
    require(isinstance(module.body[-1], ast.Expr), "Unexpected WISDM data cell ending")
    module.body.pop()  # Omit the development notebook's full-dataset inspection print.
    exec(compile(module, "<A0 data cleaning>", "exec"), context)
    exec(notebook_code(args.notebooks, "A0", 5), context)
    return context


def notebook_functions(folder, model, **context):
    module = ast.parse(notebook_code(folder, model, 7))
    module.body = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(module, f"<{model} model definitions>", "exec"), context)
    return context


def environment(np, sklearn, tf):
    return dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                scikit_learn=sklearn.__version__, tensorflow=tf.__version__, keras=tf.keras.__version__,
                gpu=[tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name)
                     for gpu in tf.config.list_physical_devices("GPU")])


def refit(args):
    selection, selection_hash = read_selection(args)
    import joblib
    import numpy as np
    import sklearn
    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier

    require(tf.config.list_physical_devices("GPU"), "A GPU is required for the final CNN refits")
    context = preparer(args, np)
    dev = {user for group in GROUPS for user in group}
    context["usable"] = [(u, a, x) for u, a, x in context["usable"] if u in dev]
    samples = np.concatenate([x for _, _, x in context["usable"]])
    mean, std = samples.mean(axis=0), samples.std(axis=0)
    require(np.isfinite(std).all() and (std > 0).all(), "Invalid development scaling")
    features = notebook_functions(args.notebooks, "RF", np=np)["features"]
    X_cnn, y, users = context["make_windows"](dev, PROTOCOL["strides"][0], mean, std)
    X_raw, y_raw, users_raw = context["make_windows"](dev, PROTOCOL["strides"][0], 0, 1)
    require(np.array_equal(y, y_raw) and np.array_equal(users, users_raw) and
            set(users) == dev and set(y) == set(range(len(ACTIVITIES))), "Invalid development windows")
    X_rf = features(X_raw)
    fit_dir = args.output / "refits"
    fit_dir.mkdir(exist_ok=True)
    metadata = fit_dir / "development.json"
    development = dict(selection_sha256=selection_hash, users=sorted(dev), windows=len(y),
                       counts_by_activity=dict(zip(ACTIVITIES, np.bincount(y).tolist())),
                       cnn_mean=mean.tolist(), cnn_std=std.tolist(), rf_features=X_rf.shape[1],
                       environment=environment(np, sklearn, tf))
    if metadata.exists():
        require(read_json(metadata) == development, "Refit environment or development inputs changed")
    else:
        save_json(metadata, development)
    specs = {row["model"]: row for row in selection["cnn_development"]}
    for name in selection["final_models"]:
        for seed in SEEDS:
            prefix = f"{name}_seed{seed}"
            record = fit_dir / f"{prefix}.json"
            if record.exists():
                check_result(record, fit_dir, selection_hash)
                continue
            started = time.perf_counter()
            if name == "RF":
                model = RandomForestClassifier(**selection["rf_params"], random_state=seed)
                model.fit(X_rf, y)
                seconds = time.perf_counter() - started
                checkpoint = fit_dir / f"{prefix}.joblib"
                joblib.dump(model, checkpoint, compress=3)
                artifacts = [checkpoint]
                epochs, params = None, None
            else:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = notebook_functions(args.notebooks, name, WIN=PROTOCOL["window"],
                                           MODEL_NAME=name)["build_model"]()
                epochs, params = specs[name]["refit_epochs"], model.count_params()
                require(params == specs[name]["params"], f"{name}: parameter count differs from development")
                model.compile(optimizer=tf.keras.optimizers.Adam(PROTOCOL["lr"]),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                history = model.fit(X_cnn, y, epochs=epochs, batch_size=PROTOCOL["batch_size"], verbose=2)
                seconds = time.perf_counter() - started
                checkpoint = fit_dir / f"{prefix}.weights.h5"
                model.save_weights(checkpoint)
                curve = fit_dir / f"{prefix}_history.csv"
                save_csv(curve, [dict(epoch=i + 1, **{key: values[i] for key, values in history.history.items()})
                                 for i in range(epochs)])
                artifacts = [checkpoint, curve]
            save_json(record, dict(model=name, seed=seed, selection_sha256=selection_hash,
                                   train_seconds=round(seconds, 3), epochs=epochs, params=params,
                                   checkpoint=checkpoint.name, artifacts={p.name: digest(p) for p in artifacts}))
            print(f"Refitted {name} seed {seed}: {seconds:.1f}s", flush=True)


def check_result(record, directory, selection_hash):
    saved = read_json(record)
    require(saved["selection_sha256"] == selection_hash and
            all(digest(directory / name) == value for name, value in saved["artifacts"].items()),
            f"Invalid saved artifacts for {record.name}")
    return saved


def evaluate(args):
    selection, selection_hash = read_selection(args)
    import joblib
    import numpy as np
    import sklearn
    import tensorflow as tf

    fit_dir = args.output / "refits"
    development = read_json(fit_dir / "development.json")
    require(development["selection_sha256"] == selection_hash, "Development metadata differs from freeze")
    fits = [check_result(fit_dir / f"{name}_seed{seed}.json", fit_dir, selection_hash)
            for name in selection["final_models"] for seed in SEEDS]
    test_dir = args.output / "holdout"
    test_dir.mkdir(exist_ok=True)
    records = [test_dir / f"{fit['model']}_seed{fit['seed']}.json" for fit in fits]
    if all(path.exists() for path in records):
        for path in records:
            check_result(path, test_dir, selection_hash)
        print("All holdout predictions are already saved and verified.", flush=True)
        return
    metadata = test_dir / "evaluation.json"
    execution = dict(selection_sha256=selection_hash, users=TEST_USERS,
                     environment=environment(np, sklearn, tf))
    if metadata.exists():
        previous = read_json(metadata)
        execution["started_at"] = previous["started_at"]
        require(execution == previous, "Holdout evaluation environment changed during recovery")
    else:
        execution["started_at"] = utc_now()
        save_json(metadata, execution)
    context = preparer(args, np)
    mean, std = (np.array(development[key], dtype=np.float32) for key in ("cnn_mean", "cnn_std"))
    X_cnn, y, users = context["make_windows"](TEST_USERS, PROTOCOL["strides"][1], mean, std)
    X_raw, y_raw, users_raw = context["make_windows"](TEST_USERS, PROTOCOL["strides"][1], 0, 1)
    require(len(y) == 1411 and np.array_equal(y, y_raw) and np.array_equal(users, users_raw) and
            set(users) == set(TEST_USERS) and set(y) == set(range(len(ACTIVITIES))), "Invalid holdout windows")
    features = notebook_functions(args.notebooks, "RF", np=np)["features"]
    X_rf = features(X_raw)
    for fit, record in zip(fits, records):
        if record.exists():
            check_result(record, test_dir, selection_hash)
            continue
        name, seed = fit["model"], fit["seed"]
        checkpoint = fit_dir / fit["checkpoint"]
        if name == "RF":
            model = joblib.load(checkpoint)
            probs = model.predict_proba(X_rf)
            pred = model.classes_[probs.argmax(axis=1)]
        else:
            tf.keras.backend.clear_session()
            model = notebook_functions(args.notebooks, name, WIN=PROTOCOL["window"],
                                       MODEL_NAME=name)["build_model"]()
            model.load_weights(checkpoint)
            probs = model.predict(X_cnn, batch_size=64, verbose=0)
            pred = probs.argmax(axis=1)
        path = test_dir / f"{name}_seed{seed}_predictions.npz"
        np.savez_compressed(path, y_true=y, y_pred=pred, users=users, probabilities=probs)
        save_json(record, dict(selection_sha256=selection_hash, model=name, seed=seed,
                               checkpoint_sha256=digest(checkpoint), artifacts={path.name: digest(path)}))
        print(f"Saved holdout predictions: {name} seed {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", type=Path, default=WEEK)
    parser.add_argument("--runs", type=Path, default=WEEK / "runs")
    parser.add_argument("--output", type=Path, default=WEEK / "final")
    parser.add_argument("--archive", type=Path,
                        default=WEEK.parent / "DATA/WISDM/WISDM_ar_latest.tar.gz")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (freeze, refit, evaluate):
        commands.add_parser(command.__name__).set_defaults(action=command)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
