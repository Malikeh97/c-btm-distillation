import argparse
import gzip
import itertools
import json
import numpy as np
import pandas as pd
import pickle
import torch
import tqdm
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
import random


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


def train_vectorizer(dataset_name, path_to_vectorizer, sample_size=20000):
    # english stopwords plus the #NUMBER dummy token
    stop_words = list(text.ENGLISH_STOP_WORDS.union(["#NUMBER"]))

    model = Pipeline([('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
                      ('svd', TruncatedSVD(n_components=100)),
                      ('normalizer', Normalizer(copy=False))])

    texts_df = load_tulu_dataset(dataset_name, sample_size=sample_size)
    vecs = model.fit_transform(tqdm(texts_df.text))
    
    with open(path_to_vectorizer, 'wb+') as f:
        _ = pickle.dump(model, f)

    return model, vecs


def train_kmeans(vecs, n_clusters, path_to_kmeans, balanced=False):
    print(f"Number of samples: {vecs.shape[0]}")
    print(f"Number of clusters requested: {n_clusters}")
    print(f"Vector dimensions: {vecs.shape[1]}")
    
    if n_clusters > vecs.shape[0]:
        raise ValueError(f"Cannot create {n_clusters} clusters with only {vecs.shape[0]} samples. "
                        f"Reduce --num-clusters to {vecs.shape[0]} or less.")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=balanced)
    kmeans.fit(torch.from_numpy(vecs), iter_limit=20)

    # batches = np.array_split(vecs, vecs.shape[0] // vecs.shape[0], axis=0) # Removed batching

    # for i, batch in tqdm(enumerate(batches)):
    #     kmeans.fit(torch.from_numpy(batch), iter_limit=20, online=True, iter_k=i)
    with open(path_to_kmeans,  'wb+') as f:
        _ = pickle.dump(kmeans, f)
    return kmeans


def main(dataset_name="allenai/tulu-3-sft-mixture",
         model='tfidf',
         n_clusters=16,
         balanced=False,
         output_dir=Path("cluster_output/"),
         kmeans_only=False,
         sample_size=20000):
    
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    path_to_vectorizer = output_dir / "tfidf.pkl"
    if not kmeans_only:
        vectorizer, vecs = train_vectorizer(dataset_name, path_to_vectorizer, sample_size)
    else:
        vectorizer = load_model(path_to_vectorizer)
        vecs, _ = vectorize_tulu_dataset(vectorizer, dataset_name, sample_size)
    
    path_to_kmeans = output_dir / "kmeans.pkl"
    kmeans = train_kmeans(vecs, n_clusters, path_to_kmeans, balanced=balanced)
    return vectorizer, kmeans


def eval(vectorizer, kmeans, dataset_name="allenai/tulu-3-sft-mixture"):
    top_terms = get_top_terms(vectorizer, kmeans)
    
    # Load a sample for evaluation
    vecs, metadata = vectorize_tulu_dataset(vectorizer, dataset_name, sample_size=10000)
    
    # Predict clusters
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    metadata['cluster'] = kmeans.predict(torch.from_numpy(vecs).to(device)).cpu().numpy()
    
    return metadata, top_terms


if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset-name', default="allenai/tulu-3-sft-mixture", type=str,
                       help='Hugging Face dataset name')
    parser.add_argument('--num-clusters', required=True, type=int)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--sample-size', type=int, default=20000, 
                       help='Number of samples to use for training')

    args = parser.parse_args()

    if not args.eval_only:
        vectorizer, kmeans = main(dataset_name=args.dataset_name,
                                n_clusters=args.num_clusters,
                                balanced=args.balanced,
                                output_dir=args.output_dir,
                                kmeans_only=False,
                                sample_size=args.sample_size)
    
    path_to_vectorizer = args.output_dir / "tfidf.pkl"
    path_to_kmeans = args.output_dir / "kmeans.pkl"
    
    vectorizer = load_model(path_to_vectorizer)
    kmeans = load_model(path_to_kmeans)
    metadata, top_terms = eval(vectorizer, kmeans, args.dataset_name)
    print(top_terms)