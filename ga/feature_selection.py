#!/usr/bin/env python3
"""Run GA, forward selection and backward elimination on Ti-PMFF descriptors."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def regressor(task, seed=42):
    if task["model"] == "ridge":
        return Ridge(alpha=task["alpha"], random_state=seed)
    return XGBRegressor(**task["xgboost"], random_state=seed)


def pipeline(task, seed=42):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("model", regressor(task, seed))])


def metrics(y, p):
    return {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p)))}


class Evaluator:
    """Evaluate feature masks."""
    def __init__(self, task, seed=42):
        self.task, self.seed = task, seed
        data = pd.read_csv(ROOT / task["data"])
        features = pd.read_csv(ROOT / task["features"])
        target = pd.to_numeric(data[task["target_column"]], errors="coerce")
        valid = target.notna() & (target > 0)
        self.rows = data.loc[valid, "source_row"].to_numpy() if "source_row" in data else data.index[valid].to_numpy()
        self.X = features.loc[valid].to_numpy(dtype=np.float64)
        self.y = target.loc[valid].to_numpy(dtype=np.float64)
        self.names = features.columns.tolist()
        prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        self.scaled = prep.fit_transform(self.X)
        self.x_mean, self.y_mean = self.scaled.mean(axis=0), self.y.mean()
        self.centered = self.scaled - self.x_mean
        self.gram = self.centered.T @ self.centered
        self.rhs = self.centered.T @ (self.y - self.y_mean)

    def score(self, mask):
        cols = np.flatnonzero(mask)
        if self.task["model"] == "ridge":
            gram = self.gram[np.ix_(cols, cols)].copy()
            gram.flat[::len(cols)+1] += self.task["alpha"]
            beta = solve(gram, self.rhs[cols], assume_a="pos", check_finite=False)
            pred = self.centered[:, cols] @ beta + self.y_mean
        else:
            model = regressor(self.task, self.seed)
            model.fit(self.scaled[:, cols], self.y)
            pred = model.predict(self.scaled[:, cols])
        return {f"train_{key}": value for key, value in metrics(self.y, pred).items()}


def fit_and_evaluate(evaluator, mask, task):
    model = pipeline(task, evaluator.seed)
    model.fit(evaluator.X[:, mask], evaluator.y)
    train_pred = model.predict(evaluator.X[:, mask])
    train = pd.DataFrame({"split": "train", "source_row": evaluator.rows,
                          "observed": evaluator.y, "predicted": train_pred})
    data = pd.read_csv(ROOT / task["test_data"])
    features = pd.read_csv(ROOT / task["test_features"])
    y = pd.to_numeric(data[task["target_column"]], errors="coerce")
    valid = y.notna() & (y > 0)
    test_pred = model.predict(features.loc[valid].to_numpy(dtype=float)[:, mask])
    test = pd.DataFrame({"split": "test", "source_row": data.loc[valid, "source_row"],
                         "observed": y.loc[valid], "predicted": test_pred})
    scores = {f"train_{key}": value for key, value in metrics(evaluator.y, train_pred).items()}
    scores.update({f"test_{key}": value for key, value in metrics(y.loc[valid], test_pred).items()})
    scores.update({"n_train": len(train), "n_test": len(test)})
    return model, scores, pd.concat([train, test], ignore_index=True)


_WORKER_EVALUATOR = None
_WORKER_THREADS = None


def init_worker(evaluator):
    global _WORKER_EVALUATOR, _WORKER_THREADS
    _WORKER_EVALUATOR = evaluator
    _WORKER_THREADS = threadpool_limits(limits=1)


def worker_score(mask):
    return _WORKER_EVALUATOR.score(mask)


class Search:
    def __init__(self, evaluator, output, pool=None):
        self.evaluator, self.output, self.pool = evaluator, output, pool
        self.records, self.cache = [], {}

    @staticmethod
    def key(mask):
        return "".join("1" if bit else "0" for bit in mask)

    def evaluate(self, masks, source, step):
        pending = {}
        for mask in masks:
            key = self.key(mask)
            if key not in self.cache:
                pending.setdefault(key, mask)
        values = self.pool.map(worker_score, pending.values(), chunksize=1) if self.pool else map(self.evaluator.score, pending.values())
        for (key, mask), value in zip(pending.items(), values):
            record = {"evaluation_id": len(self.records), "first_source": source, "first_step": step,
                      "feature_count": int(mask.sum()), "mask": key, **value}
            self.records.append(record)
            self.cache[key] = record
        return [self.cache[self.key(mask)] for mask in masks]

    @staticmethod
    def rank(record):
        # Maximize training R2; prefer fewer features when fitness ties.
        return (-record["train_r2"], record["feature_count"], record["mask"])

    def save(self):
        pd.DataFrame(self.records).to_csv(self.output / "evaluations.csv", index=False)

    def ga(self, cfg, seed):
        rng = np.random.default_rng(seed)
        dimension = self.evaluator.X.shape[1]
        size = cfg["population_size"]
        population = [np.ones(dimension, dtype=bool)]
        # Sample initial subset sizes from 1 to d.
        for _ in range(size - 1):
            mask = np.zeros(dimension, dtype=bool)
            mask[rng.choice(dimension, int(rng.integers(1, dimension + 1)), replace=False)] = True
            population.append(mask)
        best = self.evaluate([population[0]], "baseline", 0)[0]
        last_material_gain, history, requests = 0, [], []
        material_score = best["train_r2"]
        for generation in range(1, cfg["max_generations"] + 1):
            records = self.evaluate(population, "GA", generation)
            requests.extend({"generation": generation, "population_index": i, "evaluation_id": r["evaluation_id"]}
                            for i, r in enumerate(records))
            order = sorted(range(size), key=lambda i: self.rank(records[i]))
            if self.rank(records[order[0]]) < self.rank(best):
                best = records[order[0]]
            if best["train_r2"] > material_score + cfg["min_improvement"]:
                material_score, last_material_gain = best["train_r2"], generation
            history.append({"generation": generation, "best_train_r2": best["train_r2"],
                            "best_train_rmse": best["train_rmse"], "best_feature_count": best["feature_count"],
                            "best_evaluation_id": best["evaluation_id"],
                            "population_mean_train_r2": float(np.mean([r["train_r2"] for r in records])),
                            "population_unique_masks": len({r["mask"] for r in records}),
                            "evaluated_unique_masks": len(self.records),
                            "stagnant_generations": generation - last_material_gain})
            if generation == 1 or generation % 10 == 0:
                pd.DataFrame(history).to_csv(self.output / "ga_history.csv", index=False)
                print(f"{self.evaluator.task['name']} GA generation={generation} R2={best['train_r2']:.6f} k={best['feature_count']} unique={len(self.records)}", flush=True)
            if generation >= cfg["min_generations"] and generation - last_material_gain >= cfg["patience"]:
                break

            def parent():
                picks = rng.choice(size, cfg["tournament_size"], replace=False)
                return population[min(picks, key=lambda i: self.rank(records[i]))].copy()

            children = [np.array([v == "1" for v in best["mask"]])]
            children.extend(population[i].copy() for i in order[:cfg["elite_count"]-1])
            while len(children) < size:
                a, b = parent(), parent()
                if rng.random() < cfg["crossover_rate"]:
                    swap = rng.random(dimension) < .5
                    a[swap], b[swap] = b[swap].copy(), a[swap].copy()
                for child in (a, b):
                    if rng.random() < cfg["mutation_rate"]:
                        flips = rng.random(dimension) < 1 / dimension
                        if not flips.any():
                            flips[rng.integers(dimension)] = True
                        child[flips] = ~child[flips]
                    if not child.any():
                        child[rng.integers(dimension)] = True
                    children.append(child)
                    if len(children) == size:
                        break
            population = children
        pd.DataFrame(history).to_csv(self.output / "ga_history.csv", index=False)
        pd.DataFrame(requests).to_csv(self.output / "ga_population.csv", index=False)
        write_json(self.output / "ga_result.json", {**best, "generations": len(history),
                   "stop_reason": "convergence" if len(history) < cfg["max_generations"] else "max_generations"})
        self.save()
        return best

    def greedy(self, method):
        dimension = self.evaluator.X.shape[1]
        mask = np.zeros(dimension, dtype=bool) if method == "forward" else np.ones(dimension, dtype=bool)
        path, candidates = [], []
        if method == "backward":
            path.extend(self.evaluate([mask], method, 0))
        for step in range(1, dimension + 1):
            alternatives = np.flatnonzero(~mask if method == "forward" else mask)
            if method == "backward" and mask.sum() == 1:
                break
            masks = []
            for index in alternatives:
                candidate = mask.copy()
                candidate[index] = ~candidate[index]
                masks.append(candidate)
            records = self.evaluate(masks, method, step)
            chosen = min(range(len(records)), key=lambda i: self.rank(records[i]))
            candidates.extend({"step": step, "changed_feature_index": int(index),
                               "evaluation_id": r["evaluation_id"], "selected_at_step": i == chosen}
                              for i, (index, r) in enumerate(zip(alternatives, records)))
            mask = masks[chosen]
            path.append(records[chosen])
            if step == 1 or step % 16 == 0:
                print(f"{self.evaluator.task['name']} {method} step={step} k={mask.sum()} R2={path[-1]['train_r2']:.6f}", flush=True)
        pd.DataFrame(path).to_csv(self.output / f"{method}_path.csv", index=False)
        pd.DataFrame(candidates).to_csv(self.output / f"{method}_candidates.csv", index=False)
        self.save()
        return min(path, key=self.rank)


def run_task(task, config, output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    evaluator = Evaluator(task, config["seed"])
    pool = None
    try:
        if task["model"] == "xgboost" and config["workers"] > 1:
            pool = ProcessPoolExecutor(max_workers=config["workers"], mp_context=mp.get_context("fork"),
                                       initializer=init_worker, initargs=(evaluator,))
        search = Search(evaluator, output, pool)
        baseline = search.evaluate([np.ones(evaluator.X.shape[1], dtype=bool)], "baseline", 0)[0]
        ga = search.ga(config["ga"], config["seed"])
        selected = [("all_features", baseline), ("GA", ga)]
        for method in ("forward", "backward"):
            if method in config["methods"]:
                selected.append((method, search.greedy(method)))
        summary, masks, predictions = [], {}, []
        for method, record in selected:
            mask = np.array([bit == "1" for bit in record["mask"]])
            model, scores, pred = fit_and_evaluate(evaluator, mask, task)
            pred.insert(0, "method", method)
            predictions.append(pred)
            joblib.dump({"model": model, "mask": mask, "input_feature_names": evaluator.names,
                         "task": task}, output / f"{method}_model.joblib")
            masks[method] = {"selected_count": int(mask.sum()),
                             "zero_based_indices": np.flatnonzero(mask).tolist(),
                             "selected_columns": [name for name, keep in zip(evaluator.names, mask) if keep]}
            summary.append({"task": task["name"], "stage": task["stage"], "target": task["target"],
                            "model": task["model"], "method": method,
                            "feature_count": int(mask.sum()), **scores})
        pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
        pd.DataFrame(summary).to_csv(output / "metrics.csv", index=False)
        write_json(output / "selected_features.json", masks)
        print(f"{task['name']} COMPLETE: GA k={ga['feature_count']} train R2={ga['train_r2']:.6f}; seconds={time.monotonic()-started:.1f}", flush=True)
    finally:
        if pool:
            pool.shutdown()
