"""Event clustering"""
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@dataclass
class Event:
    cluster_id: int
    article_indices: List[int]
    earliest_timestamp: datetime
    coverage_count: int
    representative_title: str
    centroid: Optional[np.ndarray] = None

def cluster_embeddings(embeddings: np.ndarray, texts: List[str], timestamps: List[datetime], 
                       titles: Optional[List[str]] = None, random_state: int = 42) -> List[Event]:
    n_articles = len(embeddings)
    if titles is None:
        titles = texts
    if n_articles < 5:
        return _create_single_cluster(embeddings, texts, timestamps, titles)
    
    min_k, max_k = 2, min(10, n_articles - 1)
    best_k = _find_optimal_k(embeddings, min_k, max_k, random_state)
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    events = _create_events_from_labels(labels, embeddings, texts, timestamps, titles, kmeans.cluster_centers_)
    events.sort(key=lambda e: e.coverage_count, reverse=True)
    return events

def _find_optimal_k(embeddings: np.ndarray, min_k: int, max_k: int, random_state: int) -> int:
    best_score, best_k = -1, min_k
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k

def _create_events_from_labels(labels: np.ndarray, embeddings: np.ndarray, texts: List[str],
                                timestamps: List[datetime], titles: List[str], centroids: np.ndarray) -> List[Event]:
    events = []
    for cluster_id in np.unique(labels):
        indices = np.where(labels == cluster_id)[0].tolist()
        cluster_timestamps = [timestamps[i] for i in indices]
        earliest_idx = indices[cluster_timestamps.index(min(cluster_timestamps))]
        events.append(Event(int(cluster_id), indices, timestamps[earliest_idx], len(indices), titles[earliest_idx], centroids[cluster_id]))
    return events

def _create_single_cluster(embeddings: np.ndarray, texts: List[str], timestamps: List[datetime], titles: List[str]) -> List[Event]:
    indices = list(range(len(embeddings)))
    earliest_idx = timestamps.index(min(timestamps))
    centroid = np.mean(embeddings, axis=0)
    return [Event(0, indices, timestamps[earliest_idx], len(indices), titles[earliest_idx], centroid)]

