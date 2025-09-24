import numpy as np 
import os
import cv2 
import umap
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

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

#1: tsne
#tsne = TSNE(n_components=2)
#[REFERÊNCIA, PODE IGNORAR] class sklearn.manifold.TSNE(n_components=3, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', max_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='pca', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)

#X_red1 = tsne.fit_transform(X)
#print(X_red1.shape)

#2: PCA
pca = PCA()
#[REFERÊNCIA, PODE IGNORAR] class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None)

pca.fit(X)

#x2 = np.arange(1,641,1)
#y2 = np.cumsum(pca.explained_variance_ratio_)


#variância de 90%: x = 71.77 = 72

#pca = PCA(n_components=72)
#X_red2 = pca.fit_transform(X)

#x2 = np.arange(1,73,1)
#y2 = np.cumsum(pca.explained_variance_ratio_)


#variância de 80%: x = 29.62 = 30

#pca = PCA(n_components=30)
#X_red3 = pca.fit_transform(X)

#x2 = np.arange(1,31,1)
#y2 = np.cumsum(pca.explained_variance_ratio_)


#variância de 75%: x = 21.68 = 22


pca = PCA(n_components=22)
X_red4 = pca.fit_transform(X)

# x2 = np.arange(1,23,1)
# y2 = np.cumsum(pca.explained_variance_ratio_)


#3: UMAP
#umap_var = umap.UMAP(n_components=2)
#X_red5 = umap_var.fit_transform(X)




#dimensões: 3
#umap_var = umap.UMAP(n_components=3)
#X_red6 = umap_var.fit_transform(X)


#dimensões: 15
#umap_var = umap.UMAP(n_components=15)
#X_red7 = umap_var.fit_transform(X)



#dimensões: 55
#umap_var = umap.UMAP(n_components=55)
#X_red8 = umap_var.fit_transform(X)



#dimensões: 101
#umap_var = umap.UMAP(n_components=101)
#X_red9 = umap_var.fit_transform(X)






# # Supondo que X_red2 seja seu PCA com n_components=72
# X_red2 = pca.fit_transform(X)  # se ainda não tiver rodado

# for k in [2, 3, 4, 5]:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_red2)

#     # Visualização: se tiver mais de 2 dimensões, pega só as duas primeiras para o gráfico
#     plt.scatter(X_red2[:, 0], X_red2[:, 1], c=labels, cmap='tab10')
#     plt.title(f"K-Means com k={k}")
#     plt.show()



for k in [2, 3, 4, 5]:
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    labels = kmedoids.fit_predict(X_red2)

    plt.scatter(X_red2[:, 0], X_red2[:, 1], c=labels, cmap='tab10')
    plt.title(f"K-Medoids com k={k}")
    plt.show()




bp = 1