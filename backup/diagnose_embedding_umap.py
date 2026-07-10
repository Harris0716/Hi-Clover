#!/usr/bin/env python3
"""
Diagnose whether poor UMAP separation comes from:
1) weak 128D embedding clustering,
2) UMAP projection / parameter sensitivity,
3) or an actual evaluation-code issue.

Outputs:
- summary_metrics.csv
- sample_distance_matrix.csv
- sample_distance_matrix.pdf
- pca_2d.pdf
- umap_default.pdf
- umap_sweep_metrics.csv
- umap_best_condition.pdf
- umap_worst_condition.pdf
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score, average_precision_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness

try:
    import umap
except ImportError:
    umap = None

from HiSiNet.HiCDatasetClass import HiCDatasetDec
import HiSiNet.models as models


def parse_list(text, cast=str):
    return [cast(x.strip()) for x in str(text).split(',') if x.strip() != '']


def resolve_json_path(json_file):
    json_path = os.path.abspath(os.path.expanduser(json_file))
    if os.path.exists(json_path):
        return json_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), json_file),
        os.path.join(script_dir, json_file),
        os.path.join(os.path.dirname(script_dir), json_file),
    ]
    for c in candidates:
        c = os.path.abspath(os.path.normpath(c))
        if os.path.exists(c):
            return c

    raise FileNotFoundError(f'Cannot find config json: {json_file}')


def is_rep2_path(path):
    name = os.path.basename(path).upper()
    return any(token in name for token in ['R2', 'REP2'])


def dataset_display_sample(dataset_key, class_id, is_r2):
    rep = 'R2' if is_r2 else 'R1'

    if dataset_key == 'TCell':
        if int(class_id) == 1:
            return f'TCells Ctrl (DP) {rep}', 'Ctrl'
        return f'TCells Treat (SP) {rep}', 'Treat'

    if dataset_key == 'liver':
        if int(class_id) == 1:
            return f'Liver TAM {rep}', 'Ctrl'
        return f'Liver NIPBL {rep}', 'Treat'

    if dataset_key == 'NPC':
        if int(class_id) == 1:
            return f'NPC Ctrl {rep}', 'Ctrl'
        return f'NPC Treat (Aux) {rep}', 'Treat'

    if int(class_id) == 1:
        return f'Ctrl {rep}', 'Ctrl'
    return f'Treat {rep}', 'Treat'


def paths_for_subset(config, dataset_key, subset):
    if subset == 'train_val':
        return config[dataset_key]['training'] + config[dataset_key]['validation']
    if subset == 'test':
        return config[dataset_key]['test']
    raise ValueError(f'Unknown subset: {subset}')


def load_model(model_name, ckpt_path, device, mask=False, embedding_dim=128):
    model = getattr(models, model_name)(mask=mask, embedding_dim=embedding_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())
    model.load_state_dict(state)
    model.eval()
    return model


def collect_embeddings(model, paths, dataset_key, device, total_samples=6000, batch_size=64, seed=42):
    rng = np.random.default_rng(seed)
    embs = []
    sample_labels = []
    condition_labels = []
    source_files = []

    samples_per_file = max(1, int(total_samples) // max(1, len(paths)))

    with torch.no_grad():
        for path in paths:
            ds = HiCDatasetDec.load(path)
            indices = rng.permutation(len(ds))[:samples_per_file]
            subset = torch.utils.data.Subset(ds, indices.tolist())
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

            is_r2 = is_rep2_path(path)
            count = 0
            for batch in loader:
                x = batch[0].to(device)
                class_ids = batch[-1].cpu().numpy()
                emb = model.forward_one(x).detach().cpu().numpy()
                embs.append(emb)

                for cid in class_ids:
                    sample_name, condition_name = dataset_display_sample(dataset_key, cid, is_r2)
                    sample_labels.append(sample_name)
                    condition_labels.append(condition_name)
                    source_files.append(os.path.basename(path))

                count += len(class_ids)
                if count >= samples_per_file:
                    break

    if not embs:
        raise RuntimeError('No embeddings collected. Please check config paths and checkpoint.')

    return {
        'emb': np.vstack(embs),
        'sample': np.asarray(sample_labels),
        'condition': np.asarray(condition_labels),
        'source_file': np.asarray(source_files),
    }


def safe_silhouette(x, labels, metric='euclidean', sample_size=3000, seed=42):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) <= len(unique):
        return np.nan
    try:
        if len(labels) > sample_size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(labels), size=sample_size, replace=False)
            return float(silhouette_score(x[idx], labels[idx], metric=metric))
        return float(silhouette_score(x, labels, metric=metric))
    except Exception:
        return np.nan


def knn_purity(x, labels, k=15, metric='euclidean'):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    k_eff = min(k + 1, len(labels))
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(x)
    neigh_idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    if neigh_idx.shape[1] == 0:
        return np.nan
    same = labels[neigh_idx] == labels[:, None]
    return float(np.mean(same))


def raw_dist_metrics(raw_dist_dir, dataset_key, subset):
    path = os.path.join(raw_dist_dir, f'{dataset_key}_{subset}_raw_dist.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path)
    dist = np.asarray(data['dist']).reshape(-1)
    lbl = np.asarray(data['lbl']).reshape(-1)
    # lbl == 0: replicate, lbl == 1: condition; larger distance => condition.
    return {
        'raw_dist_file': path,
        'auroc_distance': float(roc_auc_score(lbl, dist)),
        'auprc_condition': float(average_precision_score(lbl, dist)),
        'n_pairs': int(len(lbl)),
        'n_replicate_pairs': int(np.sum(lbl == 0)),
        'n_condition_pairs': int(np.sum(lbl == 1)),
    }


def sample_distance_matrix(emb, sample_labels, max_per_sample=1200, seed=42):
    rng = np.random.default_rng(seed)
    sample_names = list(dict.fromkeys(sample_labels.tolist()))
    sampled = {}
    for s in sample_names:
        idx = np.where(sample_labels == s)[0]
        if len(idx) > max_per_sample:
            idx = rng.choice(idx, size=max_per_sample, replace=False)
        sampled[s] = emb[idx]

    mat = np.zeros((len(sample_names), len(sample_names)), dtype=float)
    for i, s1 in enumerate(sample_names):
        for j, s2 in enumerate(sample_names):
            d = pairwise_distances(sampled[s1], sampled[s2], metric='euclidean')
            if i == j and d.shape[0] > 1:
                mask = ~np.eye(d.shape[0], dtype=bool)
                mat[i, j] = float(d[mask].mean())
            else:
                mat[i, j] = float(d.mean())
    return pd.DataFrame(mat, index=sample_names, columns=sample_names)


def plot_distance_matrix(df, out_path):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(df.values, cmap='viridis')
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(df.index, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f'{df.values[i, j]:.3f}', ha='center', va='center', color='white', fontsize=7)
    ax.set_title('Sample-to-sample Mean Distance', fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Euclidean distance')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_2d(coords, condition, sample, title, out_path):
    condition_colors = {'Ctrl': '#1F77B4', 'Treat': '#D62728'}
    sample_names = list(dict.fromkeys(sample.tolist()))
    sample_colors = ['#1F77B4', '#AEC7E8', '#D62728', '#FF9896', '#2CA02C', '#98DF8A']

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    ax = axes[0]
    for cond in ['Ctrl', 'Treat']:
        idx = condition == cond
        if np.any(idx):
            ax.scatter(coords[idx, 0], coords[idx, 1], s=5, alpha=0.5, c=condition_colors[cond], label=cond, edgecolors='none')
    ax.set_title('Colored by condition', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=2)

    ax = axes[1]
    for k, s in enumerate(sample_names):
        idx = sample == s
        ax.scatter(coords[idx, 0], coords[idx, 1], s=5, alpha=0.5, c=sample_colors[k % len(sample_colors)], label=s, edgecolors='none')
    ax.set_title('Colored by sample ID', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7, markerscale=2)

    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_umap(emb, n_neighbors, min_dist, metric, seed):
    if umap is None:
        raise ImportError('umap-learn is not installed. Install with: pip install umap-learn')
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed)
    return reducer.fit_transform(emb)


def main():
    parser = argparse.ArgumentParser(description='Embedding and UMAP diagnostic script.')
    parser.add_argument('--json_file', default='Hi-Clover/config.json')
    parser.add_argument('--model_name', default='TripletLeNetBatchNormSE')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_input', default='TCell', help='Dataset key in config.json, e.g. TCell, NPC, liver')
    parser.add_argument('--subset', choices=['train_val', 'test'], default='test')
    parser.add_argument('--raw_dist_dir', default=None, help='Optional directory containing *_raw_dist.npz')
    parser.add_argument('--out_dir', default='embedding_diagnostics')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--total_samples', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--knn_k', type=int, default=15)
    parser.add_argument('--silhouette_sample_size', type=int, default=3000)
    parser.add_argument('--matrix_max_per_sample', type=int, default=1200)
    parser.add_argument('--umap_neighbors', default='15,30,50,80')
    parser.add_argument('--umap_min_dists', default='0.0,0.1,0.3')
    parser.add_argument('--umap_metrics', default='euclidean,cosine')
    parser.add_argument('--umap_seeds', default='0,1,2')
    parser.add_argument('--default_umap_neighbors', type=int, default=80)
    parser.add_argument('--default_umap_min_dist', type=float, default=0.1)
    parser.add_argument('--default_umap_metric', default='euclidean')
    parser.add_argument('--default_umap_seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = resolve_json_path(args.json_file)
    with open(json_path) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Loading model: {args.ckpt}')
    model = load_model(args.model_name, args.ckpt, device, mask=args.mask, embedding_dim=args.embedding_dim)

    paths = paths_for_subset(config, args.data_input, args.subset)
    print(f'Collecting embeddings: dataset={args.data_input}, subset={args.subset}, files={len(paths)}')
    data = collect_embeddings(model, paths, args.data_input, device, args.total_samples, args.batch_size, args.seed)

    emb = data['emb']
    sample = data['sample']
    condition = data['condition']
    print(f'Collected embeddings: {emb.shape}')

    np.savez_compressed(
        os.path.join(args.out_dir, f'{args.data_input}_{args.subset}_sampled_embeddings.npz'),
        emb=emb,
        sample=sample,
        condition=condition,
        source_file=data['source_file'],
    )

    summary = {
        'dataset': args.data_input,
        'subset': args.subset,
        'n_embeddings': int(len(emb)),
        'embedding_dim': int(emb.shape[1]),
        'silhouette_128d_condition_euclidean': safe_silhouette(emb, condition, 'euclidean', args.silhouette_sample_size, args.seed),
        'silhouette_128d_sample_euclidean': safe_silhouette(emb, sample, 'euclidean', args.silhouette_sample_size, args.seed),
        'knn_purity_condition_128d': knn_purity(emb, condition, args.knn_k, 'euclidean'),
        'knn_purity_sample_128d': knn_purity(emb, sample, args.knn_k, 'euclidean'),
    }

    if args.raw_dist_dir is not None:
        raw_metrics = raw_dist_metrics(args.raw_dist_dir, args.data_input, args.subset)
        if raw_metrics is None:
            print(f'[Warning] Raw distance file not found in {args.raw_dist_dir}')
        else:
            summary.update(raw_metrics)

    print('Computing sample-to-sample mean distance matrix...')
    dist_df = sample_distance_matrix(emb, sample, args.matrix_max_per_sample, args.seed)
    dist_df.to_csv(os.path.join(args.out_dir, 'sample_distance_matrix.csv'))
    plot_distance_matrix(dist_df, os.path.join(args.out_dir, 'sample_distance_matrix.pdf'))

    print('Running PCA...')
    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = pca.fit_transform(emb)
    summary['pca_explained_var_pc1'] = float(pca.explained_variance_ratio_[0])
    summary['pca_explained_var_pc2'] = float(pca.explained_variance_ratio_[1])
    summary['silhouette_pca_condition'] = safe_silhouette(pca_coords, condition, 'euclidean', args.silhouette_sample_size, args.seed)
    summary['silhouette_pca_sample'] = safe_silhouette(pca_coords, sample, 'euclidean', args.silhouette_sample_size, args.seed)
    plot_2d(pca_coords, condition, sample, f'PCA Projection | {args.data_input} ({args.subset})', os.path.join(args.out_dir, 'pca_2d.pdf'))

    print('Running default UMAP...')
    default_coords = run_umap(emb, args.default_umap_neighbors, args.default_umap_min_dist, args.default_umap_metric, args.default_umap_seed)
    summary['default_umap_neighbors'] = args.default_umap_neighbors
    summary['default_umap_min_dist'] = args.default_umap_min_dist
    summary['default_umap_metric'] = args.default_umap_metric
    summary['default_umap_seed'] = args.default_umap_seed
    summary['silhouette_default_umap_condition'] = safe_silhouette(default_coords, condition, 'euclidean', args.silhouette_sample_size, args.seed)
    summary['silhouette_default_umap_sample'] = safe_silhouette(default_coords, sample, 'euclidean', args.silhouette_sample_size, args.seed)
    summary['trustworthiness_default_umap'] = float(trustworthiness(emb, default_coords, n_neighbors=min(15, len(emb) - 1), metric='euclidean'))
    plot_2d(default_coords, condition, sample, f'UMAP Default | n={args.default_umap_neighbors}, min_dist={args.default_umap_min_dist}, metric={args.default_umap_metric}', os.path.join(args.out_dir, 'umap_default.pdf'))

    print('Running UMAP parameter sweep...')
    neighbors_list = parse_list(args.umap_neighbors, int)
    min_dist_list = parse_list(args.umap_min_dists, float)
    metric_list = parse_list(args.umap_metrics, str)
    seed_list = parse_list(args.umap_seeds, int)

    sweep_rows = []
    best = None
    worst = None
    for metric in metric_list:
        for n_neighbors in neighbors_list:
            for min_dist in min_dist_list:
                for seed in seed_list:
                    print(f'  UMAP metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed}')
                    coords = run_umap(emb, n_neighbors, min_dist, metric, seed)
                    row = {
                        'metric': metric,
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'seed': seed,
                        'silhouette_condition_2d': safe_silhouette(coords, condition, 'euclidean', args.silhouette_sample_size, args.seed),
                        'silhouette_sample_2d': safe_silhouette(coords, sample, 'euclidean', args.silhouette_sample_size, args.seed),
                        'trustworthiness': float(trustworthiness(emb, coords, n_neighbors=min(15, len(emb) - 1), metric='euclidean')),
                    }
                    sweep_rows.append(row)
                    score = row['silhouette_condition_2d']
                    if not np.isnan(score):
                        if best is None or score > best[0]:
                            best = (score, coords.copy(), row.copy())
                        if worst is None or score < worst[0]:
                            worst = (score, coords.copy(), row.copy())

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(args.out_dir, 'umap_sweep_metrics.csv'), index=False)

    if best is not None:
        score, coords, row = best
        plot_2d(coords, condition, sample, f"Best UMAP by condition silhouette | n={row['n_neighbors']}, min_dist={row['min_dist']}, metric={row['metric']}, seed={row['seed']}, sil={score:.4f}", os.path.join(args.out_dir, 'umap_best_condition.pdf'))
    if worst is not None:
        score, coords, row = worst
        plot_2d(coords, condition, sample, f"Worst UMAP by condition silhouette | n={row['n_neighbors']}, min_dist={row['min_dist']}, metric={row['metric']}, seed={row['seed']}, sil={score:.4f}", os.path.join(args.out_dir, 'umap_worst_condition.pdf'))

    if len(sweep_df) > 0:
        summary['umap_sweep_condition_silhouette_min'] = float(sweep_df['silhouette_condition_2d'].min())
        summary['umap_sweep_condition_silhouette_mean'] = float(sweep_df['silhouette_condition_2d'].mean())
        summary['umap_sweep_condition_silhouette_max'] = float(sweep_df['silhouette_condition_2d'].max())
        summary['umap_sweep_sample_silhouette_min'] = float(sweep_df['silhouette_sample_2d'].min())
        summary['umap_sweep_sample_silhouette_mean'] = float(sweep_df['silhouette_sample_2d'].mean())
        summary['umap_sweep_sample_silhouette_max'] = float(sweep_df['silhouette_sample_2d'].max())
        summary['umap_sweep_trustworthiness_min'] = float(sweep_df['trustworthiness'].min())
        summary['umap_sweep_trustworthiness_mean'] = float(sweep_df['trustworthiness'].mean())
        summary['umap_sweep_trustworthiness_max'] = float(sweep_df['trustworthiness'].max())

    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, 'summary_metrics.csv'), index=False)

    print('\n========== Summary =========')
    for k, v in summary.items():
        print(f'{k}: {v}')
    print('\nSaved outputs to:')
    print(os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()
