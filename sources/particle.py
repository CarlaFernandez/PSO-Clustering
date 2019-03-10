import random
from collections import defaultdict

import numpy as np
import sklearn.metrics


class Particle:
    MIN_POS = 0
    MAX_POS = 100
    MIN_VEL = 5
    MAX_VEL = 50

    def __init__(self, doc_vectors, num_clusters):
        self.centroid_vecs = []
        # dictionary of index of centroid vector and the documents it has assigned
        self.assigned = defaultdict(list)
        self.doc_vecs = doc_vectors

        for k in range(num_clusters):
            # choose a random vector among all available document vectors as cluster centroid
            chosen_vec = doc_vectors[random.randint(0, doc_vectors.shape[0] - 1)]

            self.centroid_vecs.append(chosen_vec)

        # self.position = random.randint(self.MIN_POS, self.MAX_POS)
        # self.velocity = random.randint(self.MIN_VEL, self.MAX_VEL)
        # self.best_pos = self.position
        # self.best_val = np.inf

    def assign_closest_centroid(self, distance_metric):

        # for each document vector, check the closest centroid and assign it
        for i in range(self.doc_vecs.shape[0]):
            smallest_distance = np.inf
            current_closest = []
            for j in range(len(self.centroid_vecs)):
                centroid = self.centroid_vecs[j]
                distance = self.measure_distance(centroid, self.doc_vecs[i], distance_metric)
                if distance < smallest_distance:
                    current_closest = j
                    smallest_distance = distance

            # once we have the closest centroid, assign the document to it
            self.assigned[current_closest].append(self.doc_vecs[i])

    def measure_distance(self, centroid, vector, metric):
        if metric == 'euclidean':
            return sklearn.metrics.pairwise.pairwise_distances(centroid, vector, metric="euclidean")
        elif metric == 'cosine':
            return sklearn.metrics.pairwise.pairwise_distances(centroid, vector, metric="cosine")
        else:
            return sklearn.metrics.pairwise.pairwise_distances(centroid, vector, metric="euclidean")

    def move(self):
        # the particle changes position according to its velocity and inertia

        # the particle updates current velocity values
        pass

    def calculate_fitness(self):
        pass