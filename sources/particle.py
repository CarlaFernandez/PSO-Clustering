import random
from collections import defaultdict
from scipy.spatial import distance

import numpy as np


class Particle:
    MIN_POS = 0
    MAX_POS = 1

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


        self.velocity = np.zeros(shape=(num_clusters, doc_vectors.shape[1]))
        self.fitness = np.inf
        self.own_best_fitness = self.fitness
        self.own_best_pos = self.centroid_vecs

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

            self.assigned[current_closest].append(i)


    def measure_distance(self, centroid, vector, metric):
        if metric == 'euclidean':
            return distance.euclidean(centroid, vector)
        elif metric == 'cosine':
            return np.dot(centroid, vector) / (len(centroid) * len(vector))
        else:
            self.measure_distance(centroid, vector, "cosine")


    def move(self, inertia, cognitive, social, global_best_pos):
        for i in range(len(self.centroid_vecs)):
            self.__update_velocity(i, cognitive, global_best_pos, inertia, social)

            self.__update_position(i)


    def __update_position(self, i):
        # the particle changes position according to its velocity and wraps around
        for pos in range(len(self.centroid_vecs[0])):
            updated_pos = self.centroid_vecs[i][pos] + self.velocity[i][pos]
            self.centroid_vecs[i][pos] = updated_pos

            if (self.centroid_vecs[i][pos] > self.MAX_POS):
                self.centroid_vecs[i][pos] = self.MAX_POS

            if (self.centroid_vecs[i][pos]< self.MIN_POS):
                self.centroid_vecs[i][pos] = self.MIN_POS


    def __update_velocity(self, i, cognitive, global_best_pos, inertia, social):
        # the particle updates current velocity values
        for pos in range(len(self.velocity[0])):
            this_cognitive = cognitive * random.uniform(0, 1) * (self.own_best_pos[i][pos] - self.centroid_vecs[i][pos])
            this_social = social * random.uniform(0, 1) * (global_best_pos[i][pos] - self.centroid_vecs[i][pos])
            updated_vel = inertia * self.velocity[i][pos] + this_social + this_cognitive

            self.velocity[i][pos] = updated_vel


    def calculate_fitness(self, metric):

        # if there are empty clusters
        if len(self.assigned.keys()) != len(self.centroid_vecs):
            self.fitness = np.inf

        else:
            addc = 0
            for key in self.assigned.keys():
                centroid_cum  = 0

                # for each vector assigned to the centroid
                for vector in range(len(self.assigned[key])):
                    doc_vector = self.doc_vecs[self.assigned[key][vector]].toarray()[0]
                    distance = self.measure_distance(doc_vector, self.centroid_vecs[key], metric)
                    centroid_cum += distance

                # if there were any vectors, do the average, else infinite fitness because we do not want to leave
                # an empty cluster
                if len(self.assigned[key]) != 0:
                    addc += centroid_cum / len(self.assigned[key])
                else:
                    addc += np.inf

            self.fitness = addc / len(self.assigned.keys())

        if self.fitness < self.own_best_fitness:
            self.own_best_fitness = self.fitness
            self.own_best_pos = self.centroid_vecs

        return self.fitness