"""Validation-only sequential search; freeze selections before reading test scores."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import numba
import matplotlib
import yaml
from scipy import sparse
from threadpoolctl import threadpool_limits

from src.data.download import download
from src.data.preprocess import prepare, write_json
from src.data.synthetic import generate, align_topics
from src.evaluation.metrics import split_completion, completion_score, topic_metrics
from src.models import AuthorPriorLDA, LDABaseline
from src.inference.inference import split_rhat

MODEL_NAMES = ["LDA-sklearn", "LDA-Gibbs", "AP-LDA", "AP-shuffled"]


def make_model(name: str, k: int, concentration: float, seed: int, config: dict):
    options = dict(n_topics=k, lambda_author=float(concentration), gamma=config["gamma"],
        beta=config["beta"], random_state=int(seed), n_iter=config["n_iter"],
        burn_in=config["burn_in"], thin=config["thin"], foldin_iter=config["foldin_iter"])
    if name == "LDA-sklearn":
        options.update(n_iter=config["baseline_max_iter"], burn_in=0)
        return LDABaseline(**options)
    options["author_aware"] = name != "LDA-Gibbs"
    return AuthorPriorLDA(**options)


def permute_authors(authors: list[str], seed: int) -> list[str]:
    # Permute document-level labels, not just the names of authors.
    return np.random.default_rng(seed + 811).permutation(authors).tolist()


def run_synthetic(output: Path, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        x, authors, pi, phi, _ = generate(seed=123)
        for shuffled in (False, True):
            labels = permute_authors(authors, seed) if shuffled else authors
            model = AuthorPriorLDA(n_topics=3, lambda_author=12, beta=.05,
                n_iter=400, burn_in=200, thin=5, random_state=seed).fit(x, labels)
            order = align_topics(model.topic_words_, phi)
            indices = [model.author_ids_.index(str(a)) for a in range(len(pi))]
            estimate = model.author_topics_[indices][:, order]
            rows.append({"seed": seed, "shuffled": shuffled,
                "author_mae": float(np.abs(estimate - pi).mean()),
                "phi_mae": float(np.abs(model.topic_words_[order] - phi).mean()),
                "dominant_topic_accuracy": float((estimate.argmax(1) == pi.argmax(1)).mean()),
                "entropy_ess": model.diagnostics_["entropy_ess"]})
            if seed == seeds[0] and not shuffled:
                np.savez_compressed(output / "synthetic_recovery.npz", true_pi=pi, estimated_pi=estimate,
                                    true_phi=phi, estimated_phi=model.topic_words_[order])
    result = {"label": "SYNTHETIC ONLY", "corpus_seed": 123, "rows": rows,
        "design": "Exact AP-LDA generator; 6 authors x 24 documents x 90 tokens; K=3; lambda=12; gamma=.5; beta=.05"}
    write_json(output / "synthetic.json", result)
    return result


def run_experiments(config_path: str | Path, root: str | Path = ".", resume: bool = True):
    root = Path(root).resolve()
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    cfg, data_cfg = config["experiment"], config["dataset"]
    results = root / "results"
    for folder in ("metrics", "figures", "checkpoints", "logs"):
        (results / folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(results / "logs/run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True)
    digest = hashlib.sha256(config_bytes).hexdigest()
    snapshot = results / "config_snapshot.yaml"
    if snapshot.exists() and snapshot.read_bytes() != config_bytes:
        raise ValueError("Results already belong to another config. Use --output to select a fresh directory.")
    snapshot.write_bytes(config_bytes)
    metadata = {"config_sha256": digest, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "versions": {m.__name__:m.__version__ for m in (np, scipy, sklearn, numba, matplotlib, pd, yaml)},
        "thread_policy": "BLAS limited to one thread; sklearn n_jobs=1; Numba serial kernels"}
    write_json(results / "environment.json", metadata)
    archive = download(root / "data/raw")
    matrices, records, vocab, audit = prepare(archive, data_cfg, root / "data/processed")
    logging.info("DATA %s", {s: audit['splits'][s]['documents'] for s in matrices})
    n_author = data_cfg["train_per_author"]
    train_authors = [r["author"] for r in records["train"]]
    validation_authors = [r["author"] for r in records["validation"]]
    reference = sparse.load_npz(root / "data/processed/train_full.npz")
    val_pair = split_completion(matrices["validation"], .5, 991)
    rows = []

    def run_one(name, k, concentration, seed, docs_per_author):
        key = f"{name}_K{k}_lambda{float(concentration):g}_n{docs_per_author}_s{seed}"
        record_path = results / "metrics/runs" / f"{key}.json"
        checkpoint = results / "checkpoints" / f"{key}.npz"
        if resume and record_path.exists() and checkpoint.exists():
            result = json.loads(record_path.read_text(encoding="utf-8"))
            rows.append(result)
            return result
        indices = [i for i, r in enumerate(records["train"]) if i % n_author < docs_per_author]
        x = matrices["train"][indices]
        authors = [train_authors[i] for i in indices]
        labels = permute_authors(authors, seed) if name == "AP-shuffled" else authors
        model = make_model(name, k, concentration, seed, cfg)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(x, labels)
        train_seconds = time.perf_counter() - start
        model.save(checkpoint)
        val_labels = permute_authors(validation_authors, seed + 1) if name == "AP-shuffled" else validation_authors
        score = completion_score(model, *val_pair, val_labels, seed=1201)
        quality = topic_metrics(model.topic_words_, reference, cfg["top_m"])
        row = {"key": key, "model": name, "K": k, "lambda": float(concentration),
            "seed": seed, "documents_per_author": docs_per_author,
            "validation_log_probability": score["log_probability"], "validation_perplexity": score["perplexity"],
            "npmi": quality["npmi"], "diversity": quality["diversity"], "train_seconds": train_seconds,
            "diagnostics": model.diagnostics_, "checkpoint": str(checkpoint.relative_to(root))}
        write_json(record_path, row)
        rows.append(row)
        logging.info("FIT %s val=%.5f NPMI=%.4f time=%.2fs", key, row['validation_log_probability'], row['npmi'], train_seconds)
        return row

    # First all models see an identical concentration grid at fixed K.
    for concentration in cfg["lambda_grid"]:
        for name in MODEL_NAMES:
            for seed in cfg["seeds"]:
                run_one(name, cfg["n_topics"], concentration, seed, n_author)
    frame = pd.DataFrame(rows)
    lambda_selection = {}
    for name in MODEL_NAMES:
        scores = frame[frame.model == name].groupby("lambda").validation_log_probability.mean()
        lambda_selection[name] = float(scores.idxmax())
    write_json(results / "metrics/lambda_selection.json", lambda_selection)
    # Second search K at each model's validation-selected concentration.
    visited = {row["key"] for row in rows}
    for k in cfg["topic_grid"]:
        for name in MODEL_NAMES:
            for seed in cfg["seeds"]:
                key = f"{name}_K{k}_lambda{lambda_selection[name]:g}_n{n_author}_s{seed}"
                if key not in visited:
                    run_one(name, k, lambda_selection[name], seed, n_author)
                    visited.add(key)
    frame = pd.DataFrame(rows)
    selected = {}
    for name in MODEL_NAMES:
        candidates = frame[(frame.model == name) & (frame["lambda"] == lambda_selection[name])]
        scores = candidates.groupby("K").validation_log_probability.mean()
        selected[name] = {"K": int(scores.idxmax()), "lambda": lambda_selection[name]}
    # Crucially this file is created before any heldout test scoring.
    selection = {"criterion": "Sequential mean validation plug-in log probability: lambda then K",
        "test_used_for_selection": False, "models": selected,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(results / "metrics/selection.json", selection)
    # Sparse-author analysis uses a predeclared common K/lambda for all models.
    for count in cfg["documents_per_author_grid"]:
        for name in MODEL_NAMES:
            for seed in cfg["seeds"]:
                key = f"{name}_K{cfg['n_topics']}_lambda{cfg['lambda_author']:g}_n{count}_s{seed}"
                if key not in visited:
                    run_one(name, cfg["n_topics"], cfg["lambda_author"], seed, count)
                    visited.add(key)
    frame = pd.DataFrame([{k:v for k,v in row.items() if k != "diagnostics"} for row in rows])
    frame.to_csv(results / "metrics/validation_runs.csv", index=False, encoding="utf-8-sig")
    # Test evaluation: selected settings plus explicitly predeclared matched comparison.
    test_authors = [r["author"] for r in records["test"]]
    test_rows = []
    per_document_rows = []
    for protocol in ("validation-selected", "matched-fixed"):
        for name in MODEL_NAMES:
            setting = selected[name] if protocol == "validation-selected" else {"K": cfg["n_topics"], "lambda": cfg["lambda_author"]}
            for seed in cfg["seeds"]:
                key = f"{name}_K{setting['K']}_lambda{setting['lambda']:g}_n{n_author}_s{seed}"
                record = next(row for row in rows if row["key"] == key)
                model = AuthorPriorLDA.load(root / record["checkpoint"])
                labels = permute_authors(test_authors, seed + 2) if name == "AP-shuffled" else test_authors
                for fraction in cfg["observation_fractions"]:
                    observed, held = split_completion(matrices["test"], fraction, 993)
                    score = completion_score(model, observed, held, labels, seed=1203)
                    row = {"protocol": protocol, "model": name, "seed": seed, "K": setting["K"],
                        "lambda": setting["lambda"], "observation_fraction": fraction,
                        **{k:score[k] for k in ("log_probability", "perplexity", "heldout_tokens")},
                        "npmi": record["npmi"], "diversity": record["diversity"], "train_seconds": record["train_seconds"]}
                    test_rows.append(row)
                    for i, document in enumerate(records["test"]):
                        per_document_rows.append({"protocol": protocol, "model": name, "seed": seed,
                            "fraction": fraction, "document_id": document["id"], "author": document["author"],
                            "log_sum": score["per_document_log_sum"][i], "tokens": score["per_document_token_count"][i]})
    pd.DataFrame(test_rows).to_csv(results / "metrics/test_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_document_rows).to_csv(results / "metrics/test_document_scores.csv", index=False, encoding="utf-8-sig")
    # Label-invariant entropy diagnostics for matched model chains on exactly the same data.
    chains = [AuthorPriorLDA.load(results / "checkpoints" /
        f"AP-LDA_K{cfg['n_topics']}_lambda{cfg['lambda_author']:g}_n{n_author}_s{s}.npz")
        for s in cfg["seeds"]]
    diagnostics = {"entropy_split_rhat": split_rhat([np.asarray(m.entropy_trace_)[m.burn_in:] for m in chains]),
        "warning": "Scalar entropy diagnostics do not establish topic/label mixing or full posterior convergence.",
        "chains": [m.diagnostics_ for m in chains]}
    write_json(results / "metrics/mcmc_diagnostics.json", diagnostics)
    from src.evaluation.diagnostics import run_long_chains
    logging.info("DIAGNOSTICS: validation-only chains at 10x primary length")
    run_long_chains(root, config, matrices, records)
    run_synthetic(results / "metrics", cfg["seeds"])
    from src.visualization.plots import create_figures
    from src.evaluation.report import create_report
    create_figures(root, config)
    create_report(root, config, audit, rows, test_rows, selection, diagnostics)
    logging.info("COMPLETE: %d fitted real-corpus configurations; reports/final_report.md", len(rows))
    write_json(results / "completed.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_corpus_runs": len(rows), "test_score_rows": len(test_rows), "config_sha256": digest})
