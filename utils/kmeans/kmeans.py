import numpy as np

class KMeans:
    def __init__(self, k : int):
        self.k = k
        self.centroids = None
        self.assignments = None

    def fit(self, X : np.ndarray):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False), :]
        changed = False
        while not changed:
            old_assignments = self.assignments
            self.assignments = self.assign_points(X)
            self.centroids = self.update_centroids(X, self.assignments, self.k)
            changed = not np.array_equal(old_assignments, self.assignments)

    def assign_points(self, X : np.ndarray):
        distances = np.sum((X - self.centroids)**2, axis=1)
        assignments = np.argmin(distances, axis=1)
        return assignments

    def update_centroids(self, X : np.ndarray, assignments : np.ndarray, k : int):
        centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            cluster_points = X[assignments == i]
            centroids[i, :] = np.mean(cluster_points, axis=0)
        return centroids

    def predict(self, X : np.ndarray):
        return self.assign_points(X)
