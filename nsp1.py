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

# 1: tsne
# tsne = TSNE(n_components=3, random_state=42)
# X_red1 = tsne.fit_transform(X)
# print(X_red1.shape)
# plt.figure(1)

# plt.scatter(X_red1[:,0],X_red1[:,1], edgecolors='k',c='green')
# plt.show()


#2: PCA
# pca = PCA()
# pca.fit(X)
# plt.figure(2)
# x2 = np.arange(1,641,1)
# y2 = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(x2,y2,marker='x')
# plt.grid()
# plt.show()

# variância de 90%: x = 71.77 = 72
# pca = PCA(n_components=72)
# X_red2 = pca.fit_transform(X)

# plt.figure(2)
# x2 = np.arange(1,73,1)
# y2 = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(x2,y2,marker='x')
# plt.grid()
# plt.show()


# variância de 80%: x = 29.62 = 30

# pca = PCA(n_components=30)
# X_red3 = pca.fit_transform(X)

# plt.figure(2)
# x2 = np.arange(1,31,1)
# y2 = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(x2,y2,marker='x')
# plt.grid()
# plt.show()


# variância de 75%: x = 21.68 = 22


# pca = PCA(n_components=22)
# X_red4 = pca.fit_transform(X)

# plt.figure(2)
# x2 = np.arange(1,23,1)
# y2 = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(x2,y2,marker='x')
# plt.grid()
# plt.show()


# 2D
# umap_var = umap.UMAP(n_components=2, random_state=42)
# X_red5 = umap_var.fit_transform(X)
# plt.figure(3)
# plt.scatter(X_red5[:, 0], X_red5[:, 1], c='r', edgecolor='k')
# plt.title("UMAP - 2D")
# plt.grid()
# plt.show()

# 3D
# umap_var = umap.UMAP(n_components=3, random_state=42)
# X_red6 = umap_var.fit_transform(X)
# plt.figure(4)
# plt.scatter(X_red6[:, 0], X_red6[:, 1], c='y', edgecolor='k')
# plt.title("UMAP - 3D (projeção 2D)")
# plt.grid()
# plt.show()

# # 15D
# umap_var = umap.UMAP(n_components=15, random_state=42)
# X_red7 = umap_var.fit_transform(X)
# plt.figure(5)
# plt.scatter(X_red7[:, 0], X_red7[:, 1], c='m', edgecolor='k')
# plt.title("UMAP - 15D (projeção 2D)")
# plt.grid()
# plt.show()

# # 55D
umap_var = umap.UMAP(n_components=55, random_state=42)
X_red8 = umap_var.fit_transform(X)
plt.figure(6)
plt.scatter(X_red8[:, 0], X_red8[:, 1], c='c', edgecolor='k')
plt.title("UMAP - 55D (projeção 2D)")
plt.grid()
plt.show()

# # 101D
# umap_var = umap.UMAP(n_components=101, random_state=42)
# X_red9 = umap_var.fit_transform(X)
# plt.figure(7)
# plt.scatter(X_red9[:, 0], X_red9[:, 1], c='b', edgecolor='k')
# plt.title("UMAP - 101D (projeção 2D)")
# plt.grid()
# plt.show()

######################################################
#Parte do Sidney

# Supondo que você use os dados do UMAP 2D (X_red5)
# X_cluster = X_red5  

# # Definir número de clusters (exemplo: 5)
# kmeans = KMeans(n_clusters=5, random_state=42)
# labels = kmeans.fit_predict(X_cluster)

# # Visualização
# plt.figure(figsize=(8,6))
# plt.scatter(X_cluster[:,0], X_cluster[:,1], c=labels, cmap='viridis', edgecolor='k')
# plt.title("Clusterização com K-means (UMAP 2D)")
# plt.show()

# score = silhouette_score(X_cluster, labels)
# print("Silhouette Score (K-means):", score)


# Supondo que você use os dados do UMAP 55D (X_red8)
X_cluster = X_red8 

# Definir número de clusters (exemplo: 3)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_cluster)

# Visualização
plt.figure(figsize=(8,6))
plt.scatter(X_cluster[:,0], X_cluster[:,1], c=labels, cmap='viridis', edgecolor='k')
plt.title("Clusterização com K-means (UMAP 55D)")
plt.show()

score = silhouette_score(X_cluster, labels)
print("Silhouette Score (K-means):", score)






###############################################
#           INDICE DE DUNN#


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





bp = 1