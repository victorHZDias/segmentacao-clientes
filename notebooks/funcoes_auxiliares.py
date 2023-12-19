import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def visualizar_clusters(
    dataframe,
    colunas,
    quantidade_cores,
    centroids,
    mostrar_centroids=True, 
    mostrar_pontos=False,
    coluna_clusters=None
):

    fig = plt.figure()
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection="3d")
    ax.patch.set_facecolor('black')
     
    cores = plt.cm.tab10.colors[:quantidade_cores]
    cores = ListedColormap(cores)

    x = dataframe[colunas[0]]
    y = dataframe[colunas[1]]
    z = dataframe[colunas[2]]
    
    ligar_centroids = mostrar_centroids
    ligar_pontos = mostrar_pontos
    
    for i, centroid in enumerate(centroids):
        if ligar_centroids: 
            ax.scatter(*centroid, s=500, alpha=0.5)
            ax.text(*centroid, f"{i}", fontsize=20, horizontalalignment="center", verticalalignment="center",color="white")
    
        if ligar_pontos:
            s = ax.scatter(x, y, z, c=coluna_clusters, cmap=cores)
            ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))
    
    
    ax.set_xlabel(colunas[0],color="white")
    ax.set_ylabel(colunas[1],color="white")
    ax.set_zlabel(colunas[2],color="white")
    ax.set_title("Clusters",color="white")
    params=['x','y','z']
    for i in params:
        ax.tick_params(axis=i, colors='white')    # Cor dos rótulos dos ticks no eixo X
    plt.show()
    
def graficos_elbow_silhouette (X,random_state=42,intervalo_k=(2,11)):

    os.environ["OMP_NUM_THREADS"] = "1"

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)

    elbow = {}
    silhouette = []

    k_range = range(*intervalo_k)

    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(X)
        elbow[i] = kmeans.inertia_
        
        labels = kmeans.labels_
        silhouette.append(silhouette_score(X, labels))
        
    sns.lineplot(x=list(elbow.keys()), y=list(elbow.values()), ax=axs[0])
    axs[0].set_xlabel("K")
    axs[0].set_xlabel("Inertia")
    axs[0].set_title("Elbow Method")

    sns.lineplot(x=list(k_range), y=silhouette, ax=axs[1])
    axs[1].set_xlabel("K")
    axs[1].set_xlabel("Silhouette Score")
    axs[1].set_title("Silhouette Method")

    plt.show()
    

