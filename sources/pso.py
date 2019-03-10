from particle import Particle


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


        # for every iteration until the stop condition is met
        while not self.stop_condition_met(num_iter):
            for i in range(num_particles):
                particle = self.particles[i]

                # assign each document vector to the closest centroid
                print("Finding closest centroid vectors -- Particle {0}".format(i))
                particle.assign_closest_centroid(distance_metric)

                # each particle calculates and stores its new fitness value
                particle.calculate_fitness()

                # each particle moves according the its velocity and inertia (acceleration???)
                # and updates those values accordingly
                particle.move()

            num_iter += 1

    def initialize_particles(self, num_clusters, num_particles):
        # each particle randomly chooses k different document vectors
        # from the document collection as the initial cluster centroid vectors

        for i in range(num_particles):
            print("Choosing initial clusters -- Particle {0}".format(i))
            self.particles.append(self.createParticle(self.data, num_clusters))

    def createParticle(self, doc_vectors, num_clusters):
        return Particle(doc_vectors, num_clusters)

    def stop_condition_met(self, num_iter):
        # TODO more stopping conditions
        # stop condition may be either a maximum number of iterations
        # or the difference between updates in centroid vectors (improvement in fitness value???)
        return num_iter == 1


