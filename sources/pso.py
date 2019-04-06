import numpy as np

from documentpca import DocumentPCA
from particle import Particle
from time import sleep

class PSO:
    def __init__(self, data):
        """
        Initialize Particle Swarm Optimization using the provided data
        :param data: TF-IDF matrix of document vectors
        """
        print("Initializing PSO...")
        self.data = data
        self.particles = []

    def clustering(self, num_particles=10, num_clusters=5, distance_metric='euclidean', inertia=1, cognitive=2, social=2):
        if distance_metric != 'euclidean' and distance_metric != 'cosine':
            print("Unknown distance metric '{0}', defaulting to Euclidean".format(distance_metric))

        print("Performing clustering...")

        self.__initialize_particles(num_clusters, num_particles)

        self.num_iter = 0
        global_best_fitness = np.inf
        global_best_pos = self.particles[0].centroid_vecs
        global_best_particle = self.particles[0]

        pca = DocumentPCA()

        # for every iteration until the stop condition is met
        while not self.__stop_condition_met():

            for i in range(num_particles):
                particle = self.particles[i]

                # assign each document vector to the closest centroid
                particle.assign_closest_centroids(distance_metric)

                # each particle calculates and stores its new fitness value
                particle_fitness = particle.calculate_fitness(distance_metric)

                if particle_fitness < global_best_fitness:
                    global_best_fitness = particle_fitness
                    global_best_pos = particle.centroid_vecs
                    global_best_particle = particle

                # each particle moves according the its velocity and inertia (acceleration???)
                # and updates those values accordingly
                particle.move(inertia, cognitive, social, global_best_pos)
                print("Particle: {0}, current fitness: {1}".format(i, particle_fitness))
                # for k in range(num_clusters):
                #     print(particle.own_best_pos[k])


            print("------------- Iteration: {0}, best solution: {1} -------------".format(self.num_iter, global_best_fitness))

            self.num_iter += 1


            pca.visualize_2D_pca(global_best_particle)
            sleep(0.5)


        print("Best solution found, {0} clusters with centroids:".format(num_clusters))
        for k in range(num_clusters):
            print(global_best_pos[k])
        pca.keep_open()



    def __initialize_particles(self, num_clusters, num_particles):
        # each particle randomly chooses k different document vectors
        # from the document collection as the initial cluster centroid vectors
        for i in range(num_particles):
            particle = self.__createParticle(self.data, num_clusters)
            self.particles.append(particle)

    def __createParticle(self, doc_vectors, num_clusters):
        return Particle(doc_vectors, num_clusters)

    def __stop_condition_met(self):
        return self.num_iter == 20


