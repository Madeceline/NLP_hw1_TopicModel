"""Score encoded observed/heldout BOW with an already trained model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from scipy import sparse
from src.models import AuthorPriorLDA
from src.evaluation.metrics import split_completion, completion_score
from src.data.preprocess import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--input',default='data/processed/test.npz')
    parser.add_argument('--authors',default='data/processed/test_documents.json')
    parser.add_argument('--fraction',type=float,default=.5)
    parser.add_argument('--seed',type=int,default=993)
    parser.add_argument('--output',default='results/custom_evaluation.json')
    args=parser.parse_args()
    model=AuthorPriorLDA.load(args.checkpoint)
    authors=[r['author'] for r in json.loads(Path(args.authors).read_text(encoding='utf-8-sig'))]
    observed,held=split_completion(sparse.load_npz(args.input),args.fraction,args.seed)
    score=completion_score(model,observed,held,authors,seed=1203)
    write_json(Path(args.output),score)
    print({k:score[k] for k in ['log_probability','perplexity','heldout_tokens']})


if __name__=='__main__': main()
