import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generamos dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Función para calcular la distancia euclídea
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# Inicialización de centroides
def initialize_centroids(X, k, method="random"):
    if method == "random":
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        return np.array([[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for _ in range(k)])
    elif method == "points":
        indices = np.random.choice(len(X), k, replace=False)
        return X[indices]

# K-Means desde cero
def kmeans(X, k, max_iter=100, init_method="random"):
    centroids = initialize_centroids(X, k, method=init_method)

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        labels = np.zeros(len(X), dtype=int)

        for i, point in enumerate(X):
            distances = euclidean_distance(point, centroids)
            min_distance = np.min(distances)
            closest = np.where(distances == min_distance)[0]

            if len(closest) > 1:
                closest = sorted(closest, key=lambda c: len(clusters[c]))[0]
            else:
                closest = closest[0]

            labels[i] = closest
            clusters[closest].append(point)

        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])

        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    # Calculamos la SSE
    sse = sum(np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k))

    return labels, centroids, sse

# Método del codo para encontrar el mejor k
sse_values = []
k_range = range(1, 11)

for k in k_range:
    _, _, sse = kmeans(X, k, init_method="points")
    sse_values.append(sse)

# Graficamos el método del codo
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse_values, marker='o', linestyle='-', color='b')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Suma de Errores Cuadráticos (SSE)")
plt.title("Método del Codo para Seleccionar k")
plt.show()

# Elegimos el mejor k según el codo (visualmente, k=4)
best_k = 4
labels, centroids, _ = kmeans(X, best_k, init_method="points")

# Visualización de los clusters con el mejor k
plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title(f"Clustering con K-Means (k={best_k})")
plt.show()