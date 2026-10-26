"""Export readable figures as PNG and editable SVG, with Chinese font support."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.models import AuthorPriorLDA
from src.data.preprocess import write_json

COLORS = {"LDA-sklearn":"#627d98", "LDA-Gibbs":"#c69240", "AP-LDA":"#167d8d", "AP-shuffled":"#bd6766"}


def configure_style():
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [n for n in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"] if n in fonts]
    plt.rcParams.update({"font.family": preferred + ["DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.dpi": 180, "svg.fonttype": "none"})
    return preferred


def save(fig, output: Path, name: str):
    fig.savefig(output / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plate_diagram(output: Path, author_aware: bool):
    name = "ap_lda_plate" if author_aware else "lda_plate"
    dot_source = '''digraph G {
      graph [rankdir=LR, compound=true, bgcolor="white"];
      node [shape=circle, fontname="Arial"];
      beta [label="β", shape=plaintext];
      subgraph cluster_topics {label="K topics"; phi [label="φ_k"];}
      beta -> phi;
      subgraph cluster_docs {label="D documents"; theta [label="θ_d"];
        subgraph cluster_words {label="N_d word positions"; z [label="z_dn"]; w [label="w_dn", style=filled, fillcolor=lightgray]; z -> w;}
        theta -> z;
      }
      phi -> w;
    '''
    if author_aware:
        dot_source += '''gamma [label="γ", shape=plaintext]; lambda [label="λ",shape=plaintext];
        a [label="a(d)",style=filled,fillcolor=lightgray];
        subgraph cluster_authors {label="A authors"; pi [label="π_a"];}
        gamma -> pi; pi -> theta [label="select π_a(d)"]; lambda -> theta; a -> theta; }
        '''
    else:
        dot_source += 'alpha [label="α",shape=plaintext]; alpha -> theta; }'
    (output / f"{name}.dot").write_text(dot_source, encoding="utf-8")
    dot = shutil.which("dot")
    if dot:
        for fmt in ("png", "svg"):
            subprocess.run([dot, f"-T{fmt}", str(output/f"{name}.dot"), "-o", str(output/f"{name}.{fmt}")], check=True)
        return "Graphviz"
    # A code-native drawing fallback when Graphviz's system executable is absent.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set(xlim=(0, 10), ylim=(0, 5.5))
    ax.axis("off")
    def node(x, y, label, observed=False, hyper=False):
        if not hyper:
            ax.add_patch(Circle((x, y), .31, facecolor="#d5dce2" if observed else "white", edgecolor="#283c4a", lw=1.5))
        ax.text(x, y, label, ha="center", va="center", fontsize=12)
    def plate(x, y, w, h, label):
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor="#627d98",lw=1.2))
        ax.text(x+w-.1,y+.12,label,ha="right",va="bottom",fontsize=11)
    def arrow(a, b, label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,color="#526878",shrinkA=24,shrinkB=24))
        if label:
            ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.16,label,ha="center",fontsize=9)
    plate(3.4,.7,5.8,2.4,"D 篇文档")
    plate(5.6,1.05,3.2,1.5,"每篇 N_d 个词位")
    plate(6.8,3.5,1.8,1.3,"K 个主题")
    node(7.7,4.15,r"$\phi_k$")
    node(5.8,4.15,r"$\beta$",hyper=True)
    arrow((5.8,4.15),(7.7,4.15))
    node(4.3,1.95,r"$\theta_d$")
    node(6.4,1.95,r"$z_{dn}$")
    node(7.9,1.95,r"$w_{dn}$",observed=True)
    arrow((4.3,1.95),(6.4,1.95)); arrow((6.4,1.95),(7.9,1.95)); arrow((7.7,4.15),(7.9,1.95))
    if author_aware:
        plate(.4,1.3,1.8,1.3,"A 位作者")
        node(1.3,1.95,r"$\pi_a$")
        node(1.3,3.7,r"$\gamma$",hyper=True)
        arrow((1.3,3.7),(1.3,1.95))
        arrow((1.3,1.95),(4.3,1.95),r"选择 $\pi_{a(d)}$")
        node(3.3,4.25,r"$\lambda$",hyper=True); arrow((3.3,4.25),(4.3,1.95))
        node(4.8,4.25,r"$a(d)$",observed=True); arrow((4.8,4.25),(4.3,1.95))
    else:
        node(2.1,1.95,r"$\alpha$",hyper=True); arrow((2.1,1.95),(4.3,1.95))
    ax.set_title("AP-LDA：作者先验、文档与词位" if author_aware else "标准 LDA：文档与共享主题", pad=15)
    ax.text(.5,.2,"灰色：观测变量    白色圆：潜变量    无圆框：超参数；a(d) 为给定的作者索引",fontsize=10)
    save(fig,output,name)
    return "Matplotlib fallback; Graphviz dot executable unavailable; DOT sources included"


def grouped_curve(frame, x, y, ax, xlabel, ylabel, title):
    for name, color in COLORS.items():
        sub = frame[frame.model == name]
        groups = sub.groupby(x)[y].agg(["mean", "std"])
        ax.errorbar(groups.index, groups["mean"], yerr=groups["std"], marker="o", ms=4,
                    capsize=3,label=name,color=color,lw=1.6)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=.16)


def create_figures(root: Path, config: dict):
    output, metrics = root / "results/figures", root / "results/metrics"
    output.mkdir(parents=True,exist_ok=True)
    font = configure_style()
    backends = [plate_diagram(output, flag) for flag in (False, True)]
    write_json(output/"rendering.json", {"chinese_fonts":font,"plate_backends":backends})
    frame = pd.read_csv(metrics/"validation_runs.csv")
    test = pd.read_csv(metrics/"test_scores.csv")
    cfg, n = config["experiment"], config["dataset"]["train_per_author"]
    matched = test[(test.protocol == "matched-fixed") & (test.observation_fraction == cfg["primary_observation_fraction"])]
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax, metric, label in zip(axes,["npmi","diversity","perplexity"],["文档窗口 C_NPMI ↑","主题多样性 ↑","文档完成困惑度 ↓"]):
        summary = matched.groupby("model")[metric].agg(["mean","std"]).reindex(COLORS)
        ax.bar(np.arange(4),summary["mean"],yerr=summary["std"],capsize=4,color=list(COLORS.values()))
        ax.set_xticks(range(4),COLORS,rotation=20,ha="right")
        ax.set(ylabel=label,title=label)
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("统一 K 与浓度的测试集对比：均值 ± 样本标准差（三种子）",y=1.06)
    fig.tight_layout()
    save(fig,output,"model_comparison")
    main = frame[(frame.K == cfg["n_topics"]) & (frame.documents_per_author == n)]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,metric,label in zip(axes,["validation_log_probability","npmi","diversity"],["验证集每词预测 log 概率 ↑","C_NPMI ↑","主题多样性 ↑"]):
        grouped_curve(main,"lambda",metric,ax,"Dirichlet 总浓度 λ",label,"λ 敏感性")
        ax.set_xscale("log")
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); save(fig,output,"lambda_sensitivity")
    lambda_selected=json.loads((metrics/"lambda_selection.json").read_text())
    sub=pd.concat([frame[(frame.model == name)&(frame["lambda"] == lam)&(frame.documents_per_author == n)] for name,lam in lambda_selected.items()])
    fig,ax=plt.subplots(figsize=(8,4.5))
    grouped_curve(sub,"K","validation_log_probability",ax,"主题数 K","验证集每词预测 log 概率 ↑","主题数量敏感性（每种模型的 λ 由验证集选定）")
    ax.legend();fig.tight_layout();save(fig,output,"topic_sensitivity")
    sub=frame[(frame.K == cfg["n_topics"]) & (frame["lambda"] == cfg["lambda_author"])]
    fig,ax=plt.subplots(figsize=(8,4.5))
    grouped_curve(sub,"documents_per_author","validation_log_probability",ax,"每位作者的训练文档数","验证集每词预测 log 概率 ↑","作者数据稀疏性：同一词汇表、验证集与 K/λ")
    ax.legend();fig.tight_layout();save(fig,output,"author_data_sensitivity")
    # Predeclared first seed avoids choosing the best-looking checkpoint.
    seed=cfg["seeds"][0]
    def load(name,k=None,lam=None):
        k=cfg["n_topics"] if k is None else k
        lam=cfg["lambda_author"] if lam is None else lam
        return AuthorPriorLDA.load(root/"results/checkpoints"/f"{name}_K{k}_lambda{lam:g}_n{n}_s{seed}.npz")
    ap=load("AP-LDA")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for s in cfg["seeds"]:
        chain=AuthorPriorLDA.load(root/"results/checkpoints"/f"AP-LDA_K{cfg['n_topics']}_lambda{cfg['lambda_author']:g}_n{n}_s{s}.npz")
        axes[0].plot([h["iteration"] for h in chain.history_],[h["log_joint"] for h in chain.history_],label=f"seed={s}")
    baseline=load("LDA-sklearn")
    axes[1].plot([h["iteration"] for h in baseline.history_],[h["train_perplexity"] for h in baseline.history_],color=COLORS['LDA-sklearn'],label=f"seed={seed}")
    axes[0].axvline(cfg["burn_in"],ls="--",color="gray",label="burn-in")
    axes[0].set(xlabel="Gibbs 扫描轮次",ylabel="折叠 log p(Z,W,π)",title="AP-LDA 采样轨迹（并非单调优化目标）")
    axes[1].set(xlabel="变分 EM 轮次",ylabel="训练集变分困惑度",title="sklearn LDA 训练轨迹（仅用于本模型诊断）")
    for ax in axes: ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();save(fig,output,"training_convergence")
    fig,ax=plt.subplots(figsize=(10,6))
    im=ax.imshow(ap.author_topics_,aspect="auto",cmap="YlGnBu",vmin=0,vmax=1)
    ax.set_yticks(range(len(ap.author_ids_)),ap.author_ids_)
    ax.set_xticks(range(ap.n_topics),[f"T{i}" for i in range(ap.n_topics)])
    ax.set(xlabel="主题编号",ylabel="语料作者标签",title=f"作者—主题后验均值（K={ap.n_topics}, λ={ap.lambda_author:g}, seed={seed}）")
    fig.colorbar(im,ax=ax,label="概率");fig.tight_layout();save(fig,output,"author_topic_heatmap")
    fig,axes=plt.subplots(3,4,figsize=(15,9),sharey=True)
    for ax,a,probs in zip(axes.ravel(),ap.author_ids_,ap.author_topics_):
        top=np.argsort(probs)[-5:][::-1]
        ax.bar(range(len(top)),probs[top],color=COLORS['AP-LDA'])
        ax.set_xticks(range(len(top)),[f"T{i}" for i in top]);ax.set_ylim(0,1)
        ax.set(title=a,xlabel="Top-5 主题",ylabel="概率")
    fig.suptitle("每位作者的主题偏好（训练链后验均值；不表示真实偏好标签）",y=1.02)
    fig.tight_layout();save(fig,output,"author_top_topics")
    vocab=json.loads((root/"data/processed/vocabulary.json").read_text(encoding="utf-8"))
    representative=np.argsort(ap.document_topics_.mean(0))[-6:][::-1]
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for ax,k in zip(axes.ravel(),representative):
        top=np.argsort(ap.topic_words_[k])[-10:]
        ax.barh([vocab[i] for i in top],ap.topic_words_[k,top],color=COLORS['AP-LDA'])
        ax.set(title=f"主题 T{k}",xlabel="词概率",ylabel="关键词")
    fig.suptitle("按训练文档平均占比选择六个代表性主题",y=1.01)
    fig.tight_layout();save(fig,output,"topic_keywords")
    train_records=json.loads((root/"data/processed/train_documents.json").read_text(encoding="utf-8"))
    case_authors=np.argsort(-(ap.author_topics_*np.log(ap.author_topics_)).sum(1))[:3]
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,sharey=True)
    for ax,a in zip(axes,case_authors):
        docs=[i for i,r in enumerate(train_records) if r['author']==ap.author_ids_[a]][:3]
        ax.plot(ap.author_topics_[a],lw=2.5,marker="o",label="作者 π 后验均值")
        for i in docs: ax.plot(ap.document_topics_[i],alpha=.8,label=Path(train_records[i]['id']).stem)
        ax.set(title=ap.author_ids_[a],ylabel="主题概率",ylim=(0,1));ax.legend(fontsize=8)
    axes[-1].set(xlabel="主题编号");fig.suptitle("同作者文档的主题变化：选择偏好熵最低的三位作者（训练集）",y=1.01)
    fig.tight_layout();save(fig,output,"within_author_documents")
    synthetic=np.load(metrics/"synthetic_recovery.npz")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,["true_pi","estimated_pi"],["合成真值 π","经主题标签匹配的估计 π"]):
        image=ax.imshow(synthetic[key],vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        ax.set(xlabel="主题",ylabel="合成作者",title=title)
        fig.colorbar(image,ax=ax,label="概率")
    fig.tight_layout();save(fig,output,"synthetic_recovery")
    # Export the fixed demonstration model's full distributions.
    export=root/"results/distributions"
    export.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(ap.author_topics_,index=ap.author_ids_,columns=[f"T{i}" for i in range(ap.n_topics)]).rename_axis('author').to_csv(export/"author_topics.csv",encoding="utf-8-sig")
    pd.DataFrame(ap.document_topics_,index=[r['id'] for r in train_records],columns=[f"T{i}" for i in range(ap.n_topics)]).rename_axis('document_id').to_csv(export/"training_document_topics.csv",encoding="utf-8-sig")
    pd.DataFrame(ap.topic_words_,columns=vocab).rename_axis('topic').to_csv(export/"topic_word_probabilities.csv",encoding="utf-8-sig")
    write_json(export/"topic_keywords.json",{f"T{k}":[{"word":vocab[i],"probability":float(ap.topic_words_[k,i])} for i in np.argsort(ap.topic_words_[k])[-15:][::-1]] for k in range(ap.n_topics)})
    write_json(export/"demonstration_checkpoint.json",{"model":"AP-LDA","K":ap.n_topics,"lambda":ap.lambda_author,"seed":seed,
        "selection":"Predeclared matched setting and first seed; not selected by test scores."})
