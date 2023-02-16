import intensity_normalize
import numpy as np
import cv2

np.random.seed(30)


class kmean_Cluster:
    def __init__(self, K=5, max_iterators=100):
        self.K = K
        self.max_iterators = max_iterators

        self.clusters = [[] for _ in range(self.K)] 
        self.centroids = []

    def predict(self, X):
        