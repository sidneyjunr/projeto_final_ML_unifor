import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2 
import umap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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


print(X.shape)
X = X.T

#1: tsne
#tsne = TSNE(n_components=2)
#[REFERÊNCIA, PODE IGNORAR] class sklearn.manifold.TSNE(n_components=3, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', max_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='pca', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)

#X_red1 = tsne.fit_transform(X)
#print(X_red1.shape)
#plt.figure(1)
#plt.scatter(X_red1[:,0],X_red1[:,1], edgecolors='k',c='green')
#plt.show()

#2: PCA
pca = PCA()
#[REFERÊNCIA, PODE IGNORAR] class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None)

pca.fit(X)

#plt.figure(2)
#x2 = np.arange(1,641,1)
#y2 = np.cumsum(pca.explained_variance_ratio_)
#plt.plot(x2,y2,marker='x')
#plt.grid()
#plt.show()


#variância de 90%: x = 71.77 = 72

#pca = PCA(n_components=72)
#X_red2 = pca.fit_transform(X)

#plt.figure(2)
#x2 = np.arange(1,73,1)
#y2 = np.cumsum(pca.explained_variance_ratio_)
#plt.plot(x2,y2,marker='x')
#plt.grid()
#plt.show()


#variância de 80%: x = 29.62 = 30

#pca = PCA(n_components=30)
#X_red3 = pca.fit_transform(X)

#plt.figure(2)
#x2 = np.arange(1,31,1)
#y2 = np.cumsum(pca.explained_variance_ratio_)
#plt.plot(x2,y2,marker='x')
#plt.grid()
#plt.show()


#variância de 75%: x = 21.68 = 22


pca = PCA(n_components=22)
X_red4 = pca.fit_transform(X)

plt.figure(2)
x2 = np.arange(1,23,1)
y2 = np.cumsum(pca.explained_variance_ratio_)
plt.plot(x2,y2,marker='x')
plt.grid()
plt.show()


#3: UMAP
#umap_var = umap.UMAP(n_components=2)
#X_red5 = umap_var.fit_transform(X)

#plt.figure(3)
#plt.scatter(X_red5[:,0],X_red5[:,1], c='r', edgecolor='k')
#plt.show()



#dimensões: 3
#umap_var = umap.UMAP(n_components=3)
#X_red6 = umap_var.fit_transform(X)

#plt.figure(4)
#plt.scatter(X_red6[:,0],X_red6[:,1], c='y', edgecolor='k')
#plt.show()

#dimensões: 15
#umap_var = umap.UMAP(n_components=15)
#X_red7 = umap_var.fit_transform(X)

#plt.figure(5)
#plt.scatter(X_red7[:,0],X_red7[:,1], c='m', edgecolor='k')
#plt.show()


#dimensões: 55
#umap_var = umap.UMAP(n_components=55)
#X_red8 = umap_var.fit_transform(X)

#plt.figure(6)
#plt.scatter(X_red8[:,0],X_red8[:,1], c='c', edgecolor='k')
#plt.show()


#dimensões: 101
#umap_var = umap.UMAP(n_components=101)
#X_red9 = umap_var.fit_transform(X)

#plt.figure(7)
#plt.scatter(X_red9[:,0],X_red9[:,1], c='b', edgecolor='k')
#plt.show()


bp = 1