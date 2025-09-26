import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import os
import cv2 
import umap
import matplotlib.pyplot as plt


lista_pastas = os.listdir("RecFac")
X = np.empty((128*120,0))
for pasta in lista_pastas:
        lista_imagens = os.listdir(f"RecFac\\{pasta}")
        for imagens in lista_imagens:
                img = cv2.imread(f"RecFac\\{pasta}\\{imagens}", cv2.IMREAD_GRAYSCALE)
                x = img.flatten()
                X = np.hstack((
                        X,x.reshape(len(x),1)
                ))

# print(X.shape) 
X = X.T

# Redução de dimensionalidade com UMAP - 55D
umap_var = umap.UMAP(n_components=55, random_state=42)
X_red8 = umap_var.fit_transform(X)
plt.figure(1)
plt.scatter(X_red8[:, 0], X_red8[:, 1], c='c', edgecolor='k')
plt.title("UMAP - 55D (projeção 2D)")
plt.grid()
plt.show()

######################################################
# Parte da clusterização (Sidney)

# Função para calcular índice de Dunn
def dunn_index(X, labels):
    clusters = np.unique(labels)
    # calcula dispersão (diâmetro máximo de cada cluster)
    intra_dists = []
    for c in clusters:
        pontos = X[labels == c]
        if len(pontos) > 1:
            intra_dists.append(np.max(cdist(pontos, pontos)))
    delta = max(intra_dists)  # maior dispersão interna
    
    # calcula menor distância entre clusters
    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            pontos_i = X[labels == clusters[i]]
            pontos_j = X[labels == clusters[j]]
            inter_dists.append(np.min(cdist(pontos_i, pontos_j)))
    big_delta = min(inter_dists)  # menor distância entre clusters
    
    return big_delta / delta


# Implementação manual do K-Medoids (PAM)
def kmedoids(X, k, max_iter=300, random_state=42):
    np.random.seed(random_state)
    
    # Inicializa escolhendo k pontos aleatórios como medoides
    n_samples = X.shape[0]
    medoid_idxs = np.random.choice(n_samples, k, replace=False)
    medoids = X[medoid_idxs]
    
    for _ in range(max_iter):
        # Atribui cada ponto ao medoid mais próximo
        distances = cdist(X, medoids, metric='euclidean')
        labels = np.argmin(distances, axis=1)
        
        new_medoids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue
            # escolhe o ponto do cluster que minimiza a soma das distâncias
            costs = np.sum(cdist(cluster_points, cluster_points), axis=1)
            best_point = cluster_points[np.argmin(costs)]
            new_medoids.append(best_point)
        
        new_medoids = np.array(new_medoids)
        
        # Critério de parada
        if np.allclose(medoids, new_medoids):
            break
        medoids = new_medoids
    
    return labels, medoids


# Usando os dados do UMAP 55D
X_cluster = X_red8

# Testando diferentes valores de k
k_values = range(2, 8)  # exemplo: de 2 até 7 clusters
dunn_scores_kmeans = []
sil_scores_kmeans = []
dunn_scores_kmedoids = []
sil_scores_kmedoids = []

for k in k_values:
    # K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_km = kmeans.fit_predict(X_cluster)
    sil_scores_kmeans.append(silhouette_score(X_cluster, labels_km))
    dunn_scores_kmeans.append(dunn_index(X_cluster, labels_km))
    
    # K-medoids (manual)
    labels_kmd, medoids = kmedoids(X_cluster, k)
    sil_scores_kmedoids.append(silhouette_score(X_cluster, labels_kmd))
    dunn_scores_kmedoids.append(dunn_index(X_cluster, labels_kmd))


# Plotando métricas para comparar K-means e K-medoids
plt.figure(2)
plt.plot(k_values, sil_scores_kmeans, marker='o', label='Silhouette - KMeans')
plt.plot(k_values, sil_scores_kmedoids, marker='s', label='Silhouette - KMedoids')
plt.title("Silhouette Score por número de clusters")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.legend()
plt.show()

plt.figure(3)
plt.plot(k_values, dunn_scores_kmeans, marker='o', label='Dunn - KMeans')
plt.plot(k_values, dunn_scores_kmedoids, marker='s', label='Dunn - KMedoids')
plt.title("Índice de Dunn por número de clusters")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Índice de Dunn")
plt.grid()
plt.legend()
plt.show()


# Exemplo final com o melhor k (escolha manual ou baseado nos gráficos)
k_best = 3

# K-means final
kmeans_final = KMeans(n_clusters=k_best, random_state=42)
labels_final = kmeans_final.fit_predict(X_cluster)

plt.figure(4)
plt.scatter(X_cluster[:,0], X_cluster[:,1], c=labels_final, cmap='viridis', edgecolor='k')
plt.title(f"Clusterização com K-means (k={k_best})")
plt.show()

print("Silhouette Score (K-means):", silhouette_score(X_cluster, labels_final))
print("Dunn Index (K-means):", dunn_index(X_cluster, labels_final))


# K-medoids final
labels_final_kmd, medoids_final = kmedoids(X_cluster, k_best)

plt.figure(5)
plt.scatter(X_cluster[:,0], X_cluster[:,1], c=labels_final_kmd, cmap='plasma', edgecolor='k')
plt.scatter(medoids_final[:,0], medoids_final[:,1], c='red', marker='x', s=200, label="Medoides")
plt.title(f"Clusterização com K-medoids (k={k_best})")
plt.legend()
plt.show()

print("Silhouette Score (K-medoids):", silhouette_score(X_cluster, labels_final_kmd))
print("Dunn Index (K-medoids):", dunn_index(X_cluster, labels_final_kmd))


bp = 1
