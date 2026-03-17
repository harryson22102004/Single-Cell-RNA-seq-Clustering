import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
def simulate_scrna_data(n_cells=500, n_genes=2000, n_types=5):
    np.random.seed(42)
    data=[]
    labels=[]
    for t in range(n_types):
        marker_genes=np.random.choice(n_genes,100,replace=False)
        cells=np.random.negative_binomial(1,0.5,(n_cells//n_types,n_genes)).astype(float)
        cells[:,marker_genes]*=(t+2)
        data.append(cells); labels.extend([t]*(n_cells//n_types))
    return np.vstack(data), np.array(labels)
 
def preprocess(X, min_count=1):
    X=X[:,X.sum(0)>min_count]
    X=np.log1p(X / X.sum(1,keepdims=True) * 10000)
    X=(X-X.mean(0))/(X.std(0)+1e-8)
    return X
 
def cluster_scrna(X, n_clusters=5):
    X_norm=preprocess(X)
    pca=PCA(n_components=50, random_state=42)
    X_pca=pca.fit_transform(X_norm)
    km=KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  labels=km.fit_predict(X_pca)
    sil=silhouette_score(X_pca,labels)
    return labels, sil
 
X,true_labels=simulate_scrna_data()
pred,sil=cluster_scrna(X)
print(f"Cells: {X.shape[0]}, Genes: {X.shape[1]}")
print(f"Silhouette score: {sil:.3f}")
from collections import Counter; print("Cluster sizes:", dict(Counter(pred)))
