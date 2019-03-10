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

    def clustering(self, num_particles=10, num_clusters=5, distance_metric='euclidean', inertia=1, cognitive=2, social=2, version='global'):
        if distance_metric != 'euclidean' and distance_metric != 'cosine':
            print("Unknown distance metric '{0}', defaulting to Euclidean".format(distance_metric))

        print("Performing clustering...")

        self.initialize_particles(num_clusters, num_particles)

        # TODO remove this, used as placeholder for stop condition
        num_iter = 0
        best_particle = self.particles[0]

        # for every iteration until the stop condition is met
        while not self.stop_condition_met(num_iter):
            best_fitness = np.inf
            for i in range(num_particles):
                particle = self.particles[i]

                # assign each document vector to the closest centroid
                print("Finding closest centroid vectors -- Particle {0}".format(i))
                particle.assign_closest_centroids(distance_metric)

                # each particle calculates and stores its new fitness value
                fitness = particle.calculate_fitness(distance_metric)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_position = particle.position
                    best_particle = particle
                print("Fitness value for particle {0} -- {1}".format(i, fitness))

                # each particle moves according the its velocity and inertia (acceleration???)
                # and updates those values accordingly
                new_position = particle.move(inertia, cognitive, social, best_position)
                print("\nParticle {0} is now at position {1}\n".format(i, new_position))

            num_iter += 1

        print("Execution complete. Final document cluster assignment:")
        for k, v in best_particle.assigned.items():
            print(k, v)

    def initialize_particles(self, num_clusters, num_particles):
        # each particle randomly chooses k different document vectors
        # from the document collection as the initial cluster centroid vectors

        for i in range(num_particles):
            print("Choosing initial clusters -- Particle {0}".format(i))
            particle = self.createParticle(self.data, num_clusters)
            print("New particle created with position {0} and velocity {1}"
                  .format(particle.position, particle.velocity))
            self.particles.append(particle)

    def createParticle(self, doc_vectors, num_clusters):
        return Particle(doc_vectors, num_clusters)

    def stop_condition_met(self, num_iter):
        # TODO more stopping conditions
        # stop condition may be either a maximum number of iterations
        # or the difference between updates in centroid vectors (improvement in fitness value???)
        return num_iter == 2


