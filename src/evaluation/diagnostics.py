"""Longer chains as a validation-only convergence sensitivity appendix."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from src.models import AuthorPriorLDA
from src.inference.inference import split_rhat
from src.evaluation.metrics import split_completion, completion_score
from src.data.preprocess import write_json


def run_long_chains(root: Path, config: dict, matrices: dict, records: dict):
    cfg = config['experiment']
    output = root/'results/metrics/long_chain_diagnostics.json'
    if output.exists():
        return json.loads(output.read_text(encoding='utf-8'))
    train_authors = [r['author'] for r in records['train']]
    validation_authors = [r['author'] for r in records['validation']]
    observed,held=split_completion(matrices['validation'],.5,991)
    chains,rows=[],[]
    for seed in cfg['seeds']:
        model=AuthorPriorLDA(n_topics=cfg['n_topics'],lambda_author=cfg['lambda_author'],
            gamma=cfg['gamma'],beta=cfg['beta'],random_state=seed,
            n_iter=2400,burn_in=1200,thin=40,foldin_iter=cfg['foldin_iter'])
        start=time.perf_counter()
        model.fit(matrices['train'],train_authors)
        seconds=time.perf_counter()-start
        score=completion_score(model,observed,held,validation_authors,seed=1201)
        model.save(root/'results/checkpoints_diagnostics'/f'AP-LDA_long_s{seed}.npz')
        chains.append(np.asarray(model.entropy_trace_)[model.burn_in:])
        rows.append({'seed':seed,'n_iter':2400,'burn_in':1200,'thin':40,
            'entropy_ess':model.diagnostics_['entropy_ess'],
            'validation_log_probability':score['log_probability'],'train_seconds':seconds})
    result={'design':'Validation-only diagnostic, 10x primary chain length; not used to select test settings.',
        'entropy_split_rhat':split_rhat(chains),'runs':rows,
        'limitation':'Even a good scalar R-hat cannot establish full topic posterior mixing.'}
    write_json(output,result)
    return result
