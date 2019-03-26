import random
from collections import defaultdict

import numpy as np


class Particle:
    MIN_VEL = -10
    MAX_VEL = 10
    MIN_POS = 0.1
    MAX_POS = 50

    def __init__(self, doc_vectors, num_clusters):
        self.centroid_vecs = []
        # dictionary of index of centroid vector and the documents it has assigned
        self.assigned = defaultdict(list)
        self.doc_vecs = doc_vectors

        for k in range(num_clusters):
            # choose a random vector among all available document vectors as cluster centroid
            vector_selected = random.randint(0, doc_vectors.shape[0] - 1)
            chosen_vec = doc_vectors[vector_selected].toarray()[0]
            self.centroid_vecs.append(chosen_vec)


        self.velocity = []
        self.fitness = np.inf
        self.own_best_fitness = self.fitness

        for i in range(doc_vectors.shape[1]):
            self.velocity.append(0)

        self.own_best_pos = self.centroid_vecs.copy()

    def assign_closest_centroids(self, distance_metric):
        # for each document vector, check the closest centroid and assign it
        self.assigned = defaultdict(list)

        for i in range(self.doc_vecs.shape[0]):
            doc_vector = self.doc_vecs[i].toarray()[0]
            smallest_distance = np.inf
            current_closest = []

            for j in range(len(self.centroid_vecs)):
                centroid = self.centroid_vecs[j]
                distance = self.measure_distance(centroid, doc_vector, distance_metric)
                if distance < smallest_distance:
                    current_closest = j
                    smallest_distance = distance

            self.assigned[current_closest].append(doc_vector)


    def measure_distance(self, centroid, vector, metric):
        if metric == 'euclidean':
            # return sklearn.metrics.pairwise.pairwise_distances(centroid, vector, metric="euclidean")[0][0]
            # return np.linalg.norm(abs(centroid - vector))

            return np.linalg.norm(abs(centroid - vector))
        elif metric == 'cosine':
            return np.dot(centroid, vector)/(len(centroid)*len(vector))
        else:
            return np.linalg.norm(abs(centroid - vector))


    def move(self, inertia, cognitive, social, global_best_pos):
        for i in range(len(self.centroid_vecs)):
            self.__update_velocity(i, cognitive, global_best_pos, inertia, social)

            self.__update_position(i)


    def __update_position(self, i):
        # the particle changes position according to its velocity and wraps around
        for pos in range(len(self.centroid_vecs[0])):
            self.centroid_vecs[i][pos] = self.centroid_vecs[i][pos] + self.velocity[pos]

            if (self.centroid_vecs[i][pos] > self.MAX_POS):
                self.centroid_vecs[i][pos] = self.centroid_vecs[i][pos] % self.MAX_POS

            if (self.centroid_vecs[i][pos]< self.MIN_POS):
                self.centroid_vecs[i][pos] = self.centroid_vecs[i][pos] % self.MIN_POS


    def __update_velocity(self, i, cognitive, global_best_pos, inertia, social):
        # the particle updates current velocity values
        for pos in range(len(self.velocity)):
            self.velocity[pos] =  inertia * self.velocity[pos] + \
            cognitive * random.uniform(0, 1) * (self.own_best_pos[i][pos] - self.centroid_vecs[i][pos]) + \
            social * random.uniform(0, 1) * (global_best_pos[i][pos] - self.centroid_vecs[i][pos])

            if (self.velocity[pos] > self.MAX_VEL):
                self.velocity[pos] = self.velocity[pos] % self.MAX_VEL

            if (self.velocity[pos] < self.MAX_VEL):
                self.velocity[pos] = self.velocity[pos] % self.MIN_VEL


    def calculate_fitness(self, metric):
        addc = 0
        for key in self.assigned.keys():
            centroid_cum  = 0

            # for each vector assigned to the centroid
            for vector in range(len(self.assigned[key])):
                doc_vector = self.assigned[key][vector]
                distance = self.measure_distance(doc_vector, self.centroid_vecs[key], metric)
                centroid_cum += distance

            # if there were any vectors, do the average, else 0
            if len(self.assigned[key]) != 0:
                addc += centroid_cum / len(self.assigned[key])
            else:
                addc += 0

        self.fitness = addc / len(self.assigned.keys())

        if self.fitness < self.own_best_fitness:
            self.own_best_fitness = self.fitness
            self.own_best_pos = self.centroid_vecs

        return self.fitness