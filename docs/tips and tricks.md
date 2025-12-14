# Best Practices for Topic modeling with BertTopic


## Embeding Model


## Dimensionality Reduction


## Clustering

 1. ### HDBSCAN (Hierarchical Density-Based Spatial Clustering)
       - **RECOMMENDED for topic modeling**
       - Best for: Automatic topic discovery, varying cluster sizes
       - Pros: Auto-discovers number of topics, handles noise, finds clusters of varying density
       - Cons: Can be slow on large datasets, requires parameter tuning
       - When to use: Don't know number of topics, have noisy data, default choice
       - Returns: -1 for outliers/noise points
       
2. ### K-Means
    - Best for: Known number of topics, evenly-sized clusters
    - Pros: Fast, simple, deterministic, scalable
    - Cons: Requires pre-specifying number of clusters (k), assumes spherical clusters
    - When to use: Know number of topics needed, large datasets, spherical clusters
    - Note: Every document assigned to a topic (no outliers)
    
3. ### DBSCAN (Density-Based Spatial Clustering)
    - Best for: Arbitrary shaped clusters, noise detection
    - Pros: Finds arbitrary shapes, handles noise, no need to specify k
    - Cons: Struggles with varying densities, sensitive to parameters
    - When to use: Clusters have similar density, need noise detection
    - Returns: -1 for outliers/noise points
    
4. ### Agglomerative Clustering
    - Best for: Hierarchical topic structure
    - Pros: Creates hierarchy, no assumptions about cluster shape
    - Cons: Computationally expensive, requires specifying k
    - When to use: Want hierarchical topics, small-medium datasets

### General Guidelines:
-------------------
- **Don't know # of topics**: HDBSCAN > DBSCAN
- **Know # of topics**: K-Means > Agglomerative
- **Large dataset (>50k)**: K-Means > HDBSCAN
- **Noisy data**: HDBSCAN > DBSCAN > K-Means
- **Need hierarchy**: Agglomerative
- **Need speed**: K-Means > DBSCAN > HDBSCAN > Agglomerative
- **After UMAP**: HDBSCAN (best combination)
- **After PCA**: K-Means or HDBSCAN

## Model Labeling