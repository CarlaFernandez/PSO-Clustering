from particle import Particle
import numpy as np

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

        bounds = [-100000, 100000]
        self.__initialize_particles(num_clusters, num_particles, bounds)

        # TODO remove this, used as placeholder for stop condition
        num_iter = 0
        global_best_fitness = np.inf
        global_best_particle = self.particles[0]
        global_best_pos = global_best_particle.position

        # for every iteration until the stop condition is met
        while not self.__stop_condition_met(num_iter):

            for i in range(num_particles):
                particle = self.particles[i]

                # assign each document vector to the closest centroid
                # print("Finding closest centroid vectors -- Particle {0}".format(i))
                particle.assign_closest_centroids(distance_metric)

                # each particle calculates and stores its new fitness value
                particle_fitness = particle.calculate_fitness(distance_metric)
                if particle_fitness < global_best_fitness:
                    global_best_fitness = particle_fitness
                    global_best_pos = particle.position

                # each particle moves according the its velocity and inertia (acceleration???)
                # and updates those values accordingly
                particle.move(inertia, cognitive, social, global_best_pos)

            print("Iteration: {0}, best solution: {1}".format(num_iter, global_best_fitness))

            num_iter += 1

    def __initialize_particles(self, num_clusters, num_particles, bounds):
        # each particle randomly chooses k different document vectors
        # from the document collection as the initial cluster centroid vectors
        for i in range(num_particles):
            particle = self.__createParticle(self.data, num_clusters, bounds)
            self.particles.append(particle)

    def __createParticle(self, doc_vectors, num_clusters, bounds):
        return Particle(doc_vectors, num_clusters, bounds)

    def __stop_condition_met(self, num_iter):
        # TODO more stopping conditions
        return num_iter == 10


