import numpy as np

class KMeans:
    def __init__(self, k : int, max_iters : int = 100, random_state : int = 11):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.assignments = None

    def fit(self, X : np.ndarray):
        self.X = X
        np.random.RandomState(self.random_state)
        rnd_idx = np.random.permutation(X.shape[0])
        self.centroids = X[rnd_idx[:self.k]]
        for i in range(self.max_iters):
            old_centroids = self.centroids
            self.assignments = self.assign_points(X)
            self.centroids = self.update_centroids(self.assignments)
            if np.all(old_centroids == self.centroids):
                break

    def get_distances(self, X : np.ndarray):
        distances = np.zeros((X.shape[0], self.k))
        for k_i in range(self.k):
            row_norm = np.linalg.norm(X - self.centroids[k_i, :], axis=1)
            distances[:, k_i] = np.square(row_norm)
        return distances

    def assign_points(self, X : np.ndarray):
        distances = self.get_distances(X)
        assignments = np.argmin(distances, axis=1)
        return assignments

    def update_centroids(self, assignments : np.ndarray):
        centroids = np.zeros((self.k, self.X.shape[1]))
        for k_i in range(self.k):
            cluster_points = self.X[assignments == k_i]
            centroids[k_i, :] = np.mean(cluster_points, axis=0)
        return centroids

    def compute_sse(self):
        distance = np.zeros(self.X.shape[0])
        for k in range(self.k):
            distance[self.assignments == k] = np.linalg.norm(self.X[self.assignments == k] - self.centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def compute_wcss(self):
        results = np.zeros(self.k)
        for k in range(self.k):
            Ck = self.X[self.assignments == k]
            results[k] = 0
            for i in range(0, len(Ck)-1):
                for j in range(i+1, len(Ck)):
                    results[k] += np.sum(np.square(np.subtract(Ck[i], Ck[j])))
            results[k] /= len(Ck)
        return np.sum(results)

    def predict(self, X_test : np.ndarray):
        return self.assign_points(X_test)
