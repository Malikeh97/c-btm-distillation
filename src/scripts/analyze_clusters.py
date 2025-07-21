import argparse
import gzip
import itertools
import json
import numpy as np
import pandas as pd
import pickle
import torch
import tqdm
import random
import os
from itertools import chain
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from typing import Dict
from kmeans_pytorch import KMeans as BalancedKMeans
from datasets import load_dataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from umap import UMAP
import seaborn as sns


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Call before training
set_random_seeds(42)

def load_model(path_to_model: Path):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out


def load_tulu_dataset(dataset_name="allenai/tulu-3-sft-mixture", sample_size=20000):
    """Load Tulu dataset and extract user prompts"""
    dataset = load_dataset(dataset_name, split="train")
    
    # Extract user prompts
    user_prompts = []
    for i, example in enumerate(tqdm(dataset, desc="Extracting user prompts")):
        # Find the first user message
        for message in example['messages']:
            if message.get('role') == 'user':
                user_prompts.append({
                    "id": i,
                    "text": message.get('content', ''),
                    "source": example.get('source', ''),
                    "original_id": example.get('id', '')
                })
                break
    
    # Convert to DataFrame and sample
    df = pd.DataFrame(user_prompts)
    
    # Remove empty texts
    df = df[df['text'].str.strip() != '']
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Loaded {len(df)} user prompts from Tulu dataset")
    return df


def vectorize_tulu_dataset(model, dataset_name, sample_size=10000):
    """Vectorize Tulu dataset for evaluation"""
    texts_df = load_tulu_dataset(dataset_name, sample_size=sample_size)
    vecs = model.transform(tqdm(texts_df.text))
    return vecs, texts_df


def get_top_terms(vectorizer, kmeans):
    # this will only work if you use TFIDF vectorizer (which maintains vocab)
    original_space_centroids = vectorizer['svd'].inverse_transform(kmeans.cluster_centers.cpu())
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    vocab = vectorizer['tfidf'].get_feature_names_out()
    top_terms = []
    for i in range(kmeans.n_clusters):
        terms = {}
        for j in range(10):
            terms[f'term_{j}'] = vocab[order_centroids[i, j]]
        top_terms.append(terms)
    return pd.DataFrame(top_terms)

def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))



def analyze_top_terms(vectorizer, kmeans, dataset_name):
    top_terms = get_top_terms(vectorizer, kmeans)
    
    # Print formatted top terms for each cluster
    print("=" * 80)
    print("CLUSTER ANALYSIS - TOP TERMS")
    print("=" * 80)
    
    for i in range(len(top_terms)):
        print(f"\nCluster {i} ({kmeans.cluster_centers.shape[0]} total clusters):")
        print("-" * 40)
        terms = [top_terms.iloc[i][f'term_{j}'] for j in range(10)]
        print(f"Top terms: {', '.join(terms)}")
        
        # Count samples in this cluster
        if hasattr(kmeans, 'labels_'):
            cluster_size = (kmeans.labels_ == i).sum()
            print(f"Size: {cluster_size} samples")
    
    return top_terms

def visualize_clusters_2d(vecs, cluster_labels, n_clusters, output_dir, method='tsne'):
    """Visualize clusters in 2D using t-SNE or UMAP"""
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vecs)//4))
        # For t-SNE, we need to include cluster centers in the original fit
        if hasattr(kmeans, 'cluster_centers_'):
            # Combine data points and cluster centers
            centers = kmeans.cluster_centers_.cpu().numpy()
            combined_data = np.vstack([vecs, centers])
            combined_2d = reducer.fit_transform(combined_data)
            
            # Split back into data points and centers
            vecs_2d = combined_2d[:-n_clusters]
            centers_2d = combined_2d[-n_clusters:]
        else:
            vecs_2d = reducer.fit_transform(vecs)
            centers_2d = None
    else:  # umap
        reducer = UMAP(n_components=2, random_state=42)
        vecs_2d = reducer.fit_transform(vecs)
        
        # UMAP does have transform method
        if hasattr(kmeans, 'cluster_centers_'):
            centers_2d = reducer.transform(kmeans.cluster_centers_.cpu().numpy())
        else:
            centers_2d = None
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Add cluster centers if available
    if centers_2d is not None:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
        plt.legend()
    
    plt.tight_layout()
    path_to_figure = output_dir / f'{method}_cluster_visualization_{str(n_clusters)}_clusters.png'
    plt.savefig(path_to_figure, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_labeled_clusters_2d(vecs, cluster_labels, n_clusters, output_dir, method='tsne', 
                         cluster_names=None, cluster_colors=None):
    """
    Visualize clusters in 2D using t-SNE or UMAP with custom cluster mappings
    
    Args:
        vecs: Feature vectors
        cluster_labels: Cluster assignments for each point
        n_clusters: Number of clusters
        output_dir: Directory to save visualization
        method: 'tsne' or 'umap'
        cluster_names: Dict mapping cluster_id -> name (e.g., {0: "Knowledge", 1: "Math"})
        cluster_colors: Dict mapping cluster_id -> color (e.g., {0: "#FF6B6B", 1: "#3498DB"})
    """
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vecs)//4))
        # For t-SNE, we need to include cluster centers in the original fit
        if hasattr(kmeans, 'cluster_centers_'):
            # Combine data points and cluster centers
            centers = kmeans.cluster_centers_.cpu().numpy()
            combined_data = np.vstack([vecs, centers])
            combined_2d = reducer.fit_transform(combined_data)
            
            # Split back into data points and centers
            vecs_2d = combined_2d[:-n_clusters]
            centers_2d = combined_2d[-n_clusters:]
        else:
            vecs_2d = reducer.fit_transform(vecs)
            centers_2d = None
    else:  # umap
        reducer = UMAP(n_components=2, random_state=42)
        vecs_2d = reducer.fit_transform(vecs)
        
        # UMAP does have transform method
        if hasattr(kmeans, 'cluster_centers_'):
            centers_2d = reducer.transform(kmeans.cluster_centers_.cpu().numpy())
        else:
            centers_2d = None
    
    # Create plot with larger figure for better readability
    plt.figure(figsize=(14, 10))
    
    # Plot each cluster with custom colors and labels
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_points = vecs_2d[mask]
        
        # Get name and color for this cluster
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        color = cluster_colors.get(cluster_id, f'C{cluster_id}')  # Default matplotlib colors
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=color, 
                   label=f"{name} (C{cluster_id})",
                   alpha=0.7, s=30)
    
    # Add cluster centers if available
    if centers_2d is not None:
        for i, (cx, cy) in enumerate(centers_2d):
            if i < n_clusters:
                # Plot center
                plt.scatter(cx, cy, c='black', marker='x', s=200, linewidths=3)
                
                # Add name annotation
                name = cluster_names.get(i, f"Cluster {i}")
                plt.annotate(name, (cx, cy), 
                           xytext=(10, 10), textcoords='offset points',
                           fontweight='bold', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='white', 
                                   edgecolor='black',
                                   alpha=0.9))
    
    # Styling
    plt.title(f'Dataset Cluster Analysis ({method.upper()})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    
    # Enhanced legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=11, title='Clusters', title_fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save with descriptive filename
    plt.tight_layout()
    path_to_figure = output_dir / f'labeled_cluster_{method}_visualization_{n_clusters}_clusters.png'
    plt.savefig(path_to_figure, dpi=300, bbox_inches='tight')
    
    # Print summary
    print(f"\n📊 Cluster Visualization Saved: {path_to_figure}")
    print("🎯 Cluster Distribution:")
    for cid in range(n_clusters):
        name = cluster_names.get(cid, f"Cluster {cid}")
        count = (cluster_labels == cid).sum()
        percentage = count / len(cluster_labels) * 100
        print(f"   C{cid}: {name:<25} ({count:,} samples, {percentage:.1f}%)")
    
    plt.show()


def analyze_cluster_samples(metadata, n_samples_per_cluster=5):
    """Show sample prompts from each cluster"""
    
    print("\n" + "=" * 80)
    print("CLUSTER SAMPLES ANALYSIS")
    print("=" * 80)
    
    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        
        print(f"\n{'='*20} CLUSTER {cluster_id} {'='*20}")
        print(f"Size: {len(cluster_data)} samples")
        print(f"Percentage: {len(cluster_data)/len(metadata)*100:.1f}%")
        
        # Show sample prompts
        samples = cluster_data['text'].head(n_samples_per_cluster)
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  {sample[:200]}..." if len(sample) > 200 else f"  {sample}")

def cluster_statistics(metadata):
    """Compute and display cluster statistics"""
    
    print("\n" + "=" * 80)
    print("CLUSTER STATISTICS")
    print("=" * 80)
    
    cluster_stats = []
    
    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        
        # Text length statistics
        text_lengths = cluster_data['text'].str.len()
        
        stats = {
            'cluster': cluster_id,
            'count': len(cluster_data),
            'percentage': len(cluster_data)/len(metadata)*100,
            'avg_length': text_lengths.mean(),
            'median_length': text_lengths.median(),
            'source_diversity': cluster_data['source'].nunique() if 'source' in cluster_data else 'N/A'
        }
        cluster_stats.append(stats)
    
    stats_df = pd.DataFrame(cluster_stats)
    print(stats_df.to_string(index=False))
    
    return stats_df

def analyze_silhouette_tulu(
    base_path: str,
    dataset_name: str,
    sample_size: int = 10000,
    ks: list[int] = [2, 4, 6, 10, 14, 19],
    plot_filename: str = 'tulu_silhouette_scores.png',
    save_plot: bool = True
) -> tuple[list[int], list[float]]:
    """
    Loads TF-IDF and KMeans models saved per k, vectorizes data, calculates silhouette scores,
    and plots performance across different k values.
    """
    ks_done, scores = [], []

    for k in ks:
        k_dir = os.path.join(base_path, str(k))
        tfidf_pkl = os.path.join(k_dir, 'tfidf.pkl')
        kmeans_pkl = os.path.join(k_dir, 'kmeans.pkl')
        if not os.path.exists(tfidf_pkl) or not os.path.exists(kmeans_pkl):
            print(f"⚠️ Skipping k={k}: missing pickle(s)")
            continue

        print(f"\n🔍 Processing k={k}")
        with open(tfidf_pkl, 'rb') as f:
            tfidf = pickle.load(f)
        with open(kmeans_pkl, 'rb') as f:
            kmeans = pickle.load(f)

        X, _ = vectorize_tulu_dataset(tfidf, dataset_name, sample_size=sample_size)
        X = X.toarray() if hasattr(X, 'toarray') else X
        X = X.astype(np.float64)

        if hasattr(kmeans, 'labels_') and len(kmeans.labels_) == X.shape[0]:
            labels = kmeans.labels_
        else:
            # Convert NumPy array to PyTorch tensor
            X_tensor = torch.from_numpy(X).float()
            labels = kmeans.predict(X_tensor)

        if len(np.unique(labels)) < 2:
            print(f"⚠️ Only {len(np.unique(labels))} cluster(s); skipping")
            continue

        score = silhouette_score(X, labels, metric='cosine')  # 💡 Mean silhouette score :contentReference[oaicite:6]{index=6}
        print(f"✅ k={k} → silhouette score = {score:.4f}")

        ks_done.append(k)
        scores.append(score)

    if not scores:
        print("❌ No valid silhouette scores computed.")
        return ks_done, scores

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks_done, scores, 'o-', label='Silhouette Score', color='tab:blue')
    best = ks_done[int(np.argmax(scores))]
    best_score = max(scores)
    plt.plot(best, best_score, 'r*', markersize=15, label=f'Best: k={best}')

    plt.xticks(ks_done)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Mean Silhouette Score')
    plt.title('Silhouette Score vs k for Tulu Dataset')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"📈 Plot saved to {plot_filename}")
    plt.show()

    return ks_done, scores


def analyze_davies_bouldin_tulu(
    base_path: str,
    dataset_name: str,
    sample_size: int = 10000,
    ks: list[int] = [2, 4, 6, 10, 14, 19],
    plot_filename: str = 'tulu_davies_bouldin_scores.png',
    save_plot: bool = True
) -> tuple[list[int], list[float]]:
    """
    Loads TF-IDF and KMeans models saved per k, vectorizes data, calculates Davies-Bouldin scores,
    and plots performance across different k values.
    
    Note: Lower Davies-Bouldin scores indicate better clustering.
    """
    ks_done, scores = [], []

    for k in ks:
        k_dir = os.path.join(base_path, str(k))
        tfidf_pkl = os.path.join(k_dir, 'tfidf.pkl')
        kmeans_pkl = os.path.join(k_dir, 'kmeans.pkl')
        if not os.path.exists(tfidf_pkl) or not os.path.exists(kmeans_pkl):
            print(f"⚠️ Skipping k={k}: missing pickle(s)")
            continue

        print(f"\n🔍 Processing k={k}")
        with open(tfidf_pkl, 'rb') as f:
            tfidf = pickle.load(f)
        with open(kmeans_pkl, 'rb') as f:
            kmeans = pickle.load(f)

        X, _ = vectorize_tulu_dataset(tfidf, dataset_name, sample_size=sample_size)
        X = X.toarray() if hasattr(X, 'toarray') else X
        X = X.astype(np.float64)

        if hasattr(kmeans, 'labels_') and len(kmeans.labels_) == X.shape[0]:
            labels = kmeans.labels_
        else:
            # Convert NumPy array to PyTorch tensor
            X_tensor = torch.from_numpy(X).float()
            labels = kmeans.predict(X_tensor)

        if len(np.unique(labels)) < 2:
            print(f"⚠️ Only {len(np.unique(labels))} cluster(s); skipping")
            continue

        score = davies_bouldin_score(X, labels)  # 💡 Lower is better for Davies-Bouldin
        print(f"✅ k={k} → Davies-Bouldin score = {score:.4f}")

        ks_done.append(k)
        scores.append(score)

    if not scores:
        print("❌ No valid Davies-Bouldin scores computed.")
        return ks_done, scores

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks_done, scores, 'o-', label='Davies-Bouldin Score', color='tab:orange')
    best = ks_done[int(np.argmin(scores))]  # 💡 argmin because lower is better
    best_score = min(scores)  # 💡 min because lower is better
    plt.plot(best, best_score, 'r*', markersize=15, label=f'Best: k={best}')

    plt.xticks(ks_done)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score vs k for Tulu Dataset (Lower = Better)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"📈 Plot saved to {plot_filename}")
    plt.show()

    return ks_done, scores




if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset-name', default="allenai/tulu-3-sft-mixture", type=str,
                       help='Hugging Face dataset name')
    parser.add_argument('--num-clusters', required=True, type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--sample-size', type=int, default=10000, 
                       help='Number of samples to use for training')

    args = parser.parse_args()

    path_to_vectorizer = args.output_dir / "tfidf.pkl"
    path_to_kmeans = args.output_dir / "kmeans.pkl"
    vectorizer = load_model(path_to_vectorizer)
    kmeans = load_model(path_to_kmeans)

    # Get top terms
    top_terms = get_top_terms(vectorizer, kmeans)
    print(top_terms)

    # Load a sample for evaluation
    vecs, metadata = vectorize_tulu_dataset(vectorizer, args.dataset_name, sample_size=args.sample_size)

    # Predict clusters
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    metadata['cluster'] = kmeans.predict(torch.from_numpy(vecs).to(device)).cpu().numpy()

    # Analyze top terms in each cluster
    analyze_top_terms(vectorizer, kmeans, args.dataset_name)

    # Compute and display cluster statistics
    cluster_statistics(metadata)

    # Analyze cluster samples
    analyze_cluster_samples(metadata, 5)

    # # Visualize the clusters
    visualize_clusters_2d(vecs, metadata["cluster"], args.num_clusters, args.output_dir, method='tsne')

    # # Custom mapping
    # cluster_names = {
    #     0: "Knowledge & Instruction Following",  # write, provide, detailed content
    #     1: "General Mixed Tasks",                 # mixed topics, doesn't fit core categories well
    #     2: "Math",                               # number, equation, function, mathematical problems  
    #     3: "Knowledge & Reasoning",              # answer, question, reasoning tasks
    #     4: "Coding",                            # python, function, code, programming tasks
    #     5: "Knowledge (Multilingual)"           # foreign languages, knowledge in other languages
    # }

    # custom_colors = {
    #     0: '#FF5733',  # Orange-Red
    #     1: '#33FF57',  # Green
    #     2: '#3357FF',  # Blue
    #     3: '#FF33F1',  # Pink
    #     4: '#F1FF33',  # Yellow
    #     5: '#33F1FF'   # Cyan
    # }

    # visualize_labeled_clusters_2d(vecs, metadata["cluster"], args.num_clusters, args.output_dir, 
    #                     method='tsne',
    #                     cluster_names=cluster_names, 
    #                     cluster_colors=custom_colors)


    analyze_silhouette_tulu(
        base_path='/home/ehghaghi/scratch/ehghaghi/clusters/allenai/tulu-3-sft-mixture',
        dataset_name='allenai/tulu-3-sft-mixture',
        sample_size=10000,
        ks=[2, 4, 6, 10, 14, 19, 25, 50],
        plot_filename=args.output_dir / f'tulu_silhouette_scores_{args.num_clusters}.png',
        save_plot=True)


    analyze_davies_bouldin_tulu(
        base_path='/home/ehghaghi/scratch/ehghaghi/clusters/allenai/tulu-3-sft-mixture',
         dataset_name='allenai/tulu-3-sft-mixture',
         sample_size=10000,
         ks=[2, 4, 6, 10, 14, 19, 25, 50],
         plot_filename=args.output_dir / f'tulu_davies_bouldin_scores_{args.num_clusters}.png',
         save_plot=True)



    
