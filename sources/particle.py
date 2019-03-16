import random
from collections import defaultdict
import numpy as np
import scipy
import sklearn.metrics
import operator

class Particle:
    MIN_POS = 0
    MAX_POS = 100
    MIN_VEL = 1
    MAX_VEL = 3

    def __init__(self, doc_vectors, num_clusters, pos_bounds):
        self.centroid_vecs = []
        # dictionary of index of centroid vector and the documents it has assigned
        self.assigned = defaultdict(list)
        self.doc_vecs = doc_vectors
        self.MIN_POS = pos_bounds[0]
        self.MAX_POS = pos_bounds[1]

        for k in range(num_clusters):
            # choose a random vector among all available document vectors as cluster centroid
            vector_selected = random.randint(0, doc_vectors.shape[0] - 1)
            chosen_vec = doc_vectors[vector_selected]

            self.centroid_vecs.append(chosen_vec)


        self.velocity = []
        self.position = []
        self.fitness = np.inf
        self.own_best_pos = self.position.copy()
        self.own_best_fitness = self.fitness

        for i in range(doc_vectors.shape[0]):
            self.velocity.append(random.uniform(-1, 1))
            self.position.append(random.uniform(0, doc_vectors.shape[0]))


    def assign_closest_centroids(self, distance_metric):
        # for each document vector, check the closest centroid and assign it
        for i in range(self.doc_vecs.shape[0]):
            doc_vector = self.doc_vecs[i]
            smallest_distance = np.inf
            current_closest = []

            for j in range(len(self.centroid_vecs)):
                centroid = self.centroid_vecs[j]
                distance = self.measure_distance(centroid, doc_vector, distance_metric)
                if distance < smallest_distance:
                    current_closest = j
                    smallest_distance = distance

            # once we have the closest centroid, assign the document to it if it is not the centroid
            self.assigned[current_closest].append(doc_vector)


    def measure_distance(self, centroid, vector, metric):
        if metric == 'euclidean':
            return sklearn.metrics.pairwise.pairwise_distances(centroid, vector, metric="euclidean")[0][0]
        elif metric == 'cosine':
            return sklearn.metrics.pairwise.pairwise_distances(centroid, vector, metric="cosine")[0][0]
        else:
            return sklearn.metrics.pairwise.pairwise_distances(centroid, vector, metric="euclidean")[0][0]


    def move(self, inertia, cognitive, social, global_best_pos):
        for i in range(self.doc_vecs.shape[0]):
            self.__update_velocity(cognitive, global_best_pos, i, inertia, social)

            self.__update_position(i)

        return self.position


    def __update_position(self, i):
        # the particle changes position according to its velocity and wraps around
        self.position[i] = self.position[i] + self.velocity[i]

        if (self.position[i] > self.MAX_POS):
            self.position[i] = self.position[i] % self.MAX_POS

        if (self.position[i] < self.MIN_POS):
            self.position[i] = self.position[i] % self.MIN_POS


    def __update_velocity(self, cognitive, global_best_pos, i, inertia, social):
        # the particle updates current velocity values
        self.velocity[i] = inertia * self.velocity[i] + \
                           cognitive * random.uniform(0, 1) * abs(self.own_best_pos[i] - self.position[i]) + \
                           social * random.uniform(0, 1) * abs(global_best_pos[i] - self.position[i])

        if (self.velocity[i] > self.MAX_VEL):
            self.velocity[i] = self.velocity[i] % self.MAX_VEL



    def calculate_fitness(self, metric):
        addc = 0
        for key in range(len(self.assigned.keys())):
            centroid_cum  = 0

            # for each vector assigned to the centroid
            for value in range(len(self.assigned[key])):
                vector = self.assigned[key][0]
                centroid_cum += self.measure_distance(vector, self.centroid_vecs[key], metric)

            # if there were any vectors, do the average, else 0
            if len(self.assigned[key]) != 0:
                addc += centroid_cum / len(self.assigned[key])
            else:
                addc += 0

        self.fitness = addc / len(self.assigned.keys())

        if self.fitness < self.own_best_fitness:
            self.own_best_fitness = self.fitness
            self.own_best_pos = self.position.copy()

        return self.fitness