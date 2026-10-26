"""Train encoded BOW, or your own author/text JSONL, and export distributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from threadpoolctl import threadpool_limits
from src.data.preprocess import write_json, clean_text
from src.evaluation.experiments import make_model, permute_authors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--input',default='data/processed/train.npz')
    parser.add_argument('--authors',default='data/processed/train_documents.json')
    parser.add_argument('--jsonl',help='Alternative: one {author,text,id?} JSON record per line')
    parser.add_argument('--output',default='results/custom_training')
    parser.add_argument('--model',choices=['AP-LDA','LDA-sklearn','LDA-Gibbs','AP-shuffled'],default='AP-LDA')
    parser.add_argument('--topics',type=int)
    parser.add_argument('--lambda-author',type=float)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    cfg=config['experiment']
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if args.jsonl:
        rows=[json.loads(line) for line in Path(args.jsonl).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if any('author' not in r or 'text' not in r for r in rows):
            raise ValueError('Each JSONL record must include author and text')
        vectorizer=CountVectorizer(stop_words='english',min_df=config['dataset']['min_df'],
            max_df=config['dataset']['max_df'],max_features=config['dataset']['max_features'],
            token_pattern=r'(?u)\b[a-z][a-z]{2,}\b')
        x=vectorizer.fit_transform([clean_text(r['text']) for r in rows])
        write_json(output/'vectorizer.json',{'vocabulary':vectorizer.vocabulary_,'stop_words':'english'})
        vocab=vectorizer.get_feature_names_out().tolist()
    else:
        x=sparse.load_npz(args.input)
        rows=json.loads(Path(args.authors).read_text(encoding='utf-8-sig'))
        vocabulary_path=Path(args.input).parent/'vocabulary.json'
        vocab=json.loads(vocabulary_path.read_text(encoding='utf-8')) if vocabulary_path.exists() else [str(i) for i in range(x.shape[1])]
    authors=[str(r['author']) for r in rows]
    if args.model=='AP-shuffled': authors=permute_authors(authors,args.seed)
    model=make_model(args.model,args.topics or cfg['n_topics'],
        args.lambda_author if args.lambda_author is not None else cfg['lambda_author'],args.seed,cfg)
    with threadpool_limits(limits=1): model.fit(x,authors)
    model.save(output/'model.npz')
    columns=[f'T{i}' for i in range(model.n_topics)]
    pd.DataFrame(model.author_topics_,index=model.author_ids_,columns=columns).rename_axis('author').to_csv(output/'author_topics.csv',encoding='utf-8-sig')
    pd.DataFrame(model.document_topics_,index=[r.get('id',str(i)) for i,r in enumerate(rows)],columns=columns).rename_axis('document_id').to_csv(output/'document_topics.csv',encoding='utf-8-sig')
    write_json(output/'topic_keywords.json',{f'T{k}':[vocab[i] for i in np.argsort(model.topic_words_[k])[-15:][::-1]] for k in range(model.n_topics)})
    write_json(output/'training_metadata.json',{'model':args.model,'parameters':model.get_params(),'diagnostics':model.diagnostics_,'arguments':vars(args)})
    print(f'Saved checkpoint, author/document distributions and keywords to {output.resolve()}')


if __name__=='__main__': main()
