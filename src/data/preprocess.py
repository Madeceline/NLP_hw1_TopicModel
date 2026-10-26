"""Official train/test split, author-stratified validation, train-only vocabulary."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfTransformer


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert), encoding="utf-8")


def clean_text(text: str) -> str:
    text = re.sub(r"(?im)^\s*by\s+[^\n]+", " ", text)
    text = re.sub(r"\S+@\S+|https?://\S+", " ", text)
    return text.lower()


def bounded_counts(vectorizer, texts: list[str], max_tokens: int, seed: int):
    matrix = vectorizer.transform(texts).tocsr()
    rng = np.random.default_rng(seed)
    data, cols, offsets = [], [], [0]
    for d in range(matrix.shape[0]):
        row = matrix.getrow(d)
        tokens = np.repeat(row.indices, row.data)
        if len(tokens) > max_tokens:
            tokens = rng.choice(tokens, max_tokens, replace=False)
        ids, counts = np.unique(tokens, return_counts=True)
        cols.extend(ids.tolist())
        data.extend(counts.tolist())
        offsets.append(len(data))
    return sparse.csr_matrix((data, cols, offsets), shape=matrix.shape, dtype=np.int64), matrix


def prepare(archive_path: Path, config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["split_seed"])
    with ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".txt") and len(n.split("/")) == 3]
        available = sorted({n.split("/")[1] for n in names})
        authors = sorted(rng.choice(available, config["n_authors"], replace=False).tolist())
        records = {"train": [], "validation": [], "test": []}
        seen, duplicate_count = set(), 0
        for a in authors:
            groups = {}
            for original in ("C50train", "C50test"):
                candidates = sorted(n for n in names if n.startswith(f"{original}/{a}/"))
                rng.shuffle(candidates)
                groups[original] = []
                for name in candidates:
                    text = clean_text(archive.read(name).decode("utf-8", errors="replace"))
                    digest = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()
                    if digest in seen:
                        duplicate_count += 1
                        continue
                    seen.add(digest)
                    groups[original].append({"id": name, "author": a, "text": text, "sha256": digest})
            nt, nv, ne = (config[k] for k in ("train_per_author", "validation_per_author", "test_per_author"))
            if len(groups["C50train"]) < nt + nv or len(groups["C50test"]) < ne:
                raise ValueError(f"Not enough unique documents for {a}")
            records["train"].extend(groups["C50train"][:nt])
            records["validation"].extend(groups["C50train"][nt:nt + nv])
            records["test"].extend(groups["C50test"][:ne])
    # Remove reporters' names as potential signature shortcuts; retain substantive terms.
    name_words = {part.lower() for a in authors for part in re.findall(r"[A-Z][a-z]*", a)}
    stops = sorted(set(ENGLISH_STOP_WORDS) | name_words | {"reuters", "said", "say", "says"})
    vectorizer = CountVectorizer(lowercase=True, stop_words=stops,
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b", min_df=config["min_df"],
        max_df=config["max_df"], max_features=config["max_features"])
    vectorizer.fit([r["text"] for r in records["train"]])
    vocab = vectorizer.get_feature_names_out().tolist()
    write_json(output / "vectorizer.json", {"vocabulary": vectorizer.vocabulary_, "stop_words": stops})
    write_json(output / "vocabulary.json", vocab)
    matrices, full, stats = {}, {}, {}
    analyzer = vectorizer.build_analyzer()
    for i, (split, rows) in enumerate(records.items()):
        texts = [r["text"] for r in rows]
        matrix, unbounded = bounded_counts(vectorizer, texts, config["max_tokens"], config["split_seed"] + i)
        if (matrix.sum(1).A.ravel() < 2).any():
            raise ValueError(f"Too short document in {split}; inspect preprocessing")
        matrices[split], full[split] = matrix, unbounded
        sparse.save_npz(output / f"{split}.npz", matrix)
        # Untruncated train BOW is a fixed external reference for document-level NPMI.
        sparse.save_npz(output / f"{split}_full.npz", unbounded)
        write_json(output / f"{split}_documents.json", rows)
        all_tokens = [token for text in texts for token in analyzer(text)]
        stats[split] = {"documents": len(rows), "tokens_after_cap": int(matrix.sum()),
            "tokens_in_vocabulary_before_cap": int(unbounded.sum()),
            "oov_rate": sum(t not in vectorizer.vocabulary_ for t in all_tokens) / max(1, len(all_tokens)),
            "documents_per_author": dict(Counter(r["author"] for r in rows))}
    tfidf = TfidfTransformer().fit(full["train"])
    similarities = (tfidf.transform(full["test"]) @ tfidf.transform(full["train"]).T).toarray().max(1)
    summary = {"dataset": "UCI Reuter_50_50", "authors": authors, "vocabulary_size": len(vocab),
        "discarded_exact_duplicates_in_selected_authors": duplicate_count, "splits": stats,
        "test_train_tfidf_max_similarity_quantiles": np.quantile(similarities, [0.5, 0.9, 0.99, 1]).tolist(),
        "test_documents_similarity_above_095": int((similarities > .95).sum()),
        "single_author_policy": "Use corpus's provided single reporter folder; no coauthor attribution inferred.",
        "selection_policy": "Seeded random authors; official test held out; validation from official training only.",
        "token_cap_policy": "Uniform subsample without replacement of in-vocabulary word positions, independently per document.",
        "config": config}
    write_json(output / "audit.json", summary)
    return matrices, records, vocab, summary


def load_prepared(root: Path):
    matrices = {s: sparse.load_npz(root / f"{s}.npz") for s in ("train", "validation", "test")}
    records = {s: json.loads((root / f"{s}_documents.json").read_text(encoding="utf-8")) for s in matrices}
    return matrices, records, json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
