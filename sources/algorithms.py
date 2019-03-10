import random

class PSO:
    def __init__(self, data):
        """
        Initialize Particle Swarm Optimization using the provided data
        :param data: TF-IDF matrix of document vectors
        """
        print("Initializing PSO...")
        self.data = data
        self.particles = []

    def clustering(self, num_particles=10, num_clusters=5, inertia=1, cognitive=2, social=2, version='global'):
        print("Performing clustering...")

        print("Choosing initial cluster centroids")
        # each particle randomly chooses k different document vectors
        # from the document collection as the initial cluster centroid vectors
        for i in range(num_particles):
            print("Particle {0}".format(i))
            self.particles.append(self.initializeParticle(self.data, num_clusters))

        while not self.check_stop_condition():
            for i in range(num_particles):
                # each particle assigns each document vector to the closest centroid vector

                # each particle calculates and stores its new fitness value

                # each particle moves according the its velocity and inertia (acceleration???)
                # and updates those values accordingly
                continue

    def initializeParticle(self, doc_vectors, num_clusters):
        return Particle(doc_vectors, num_clusters)

    def check_stop_condition(self):
        # stop condition may be either a maximum number of iterations
        # or the difference between updates in centroid vectors (improvement in fitness value???)
        return True


class Particle:
    MIN_POS = 0
    MAX_POS = 100
    MIN_VEL = 5
    MAX_VEL = 50

    def __init__(self, doc_vectors, num_clusters):
        self.centroid_vecs = []

        for k in range(num_clusters):
            # choose a random vector among all available document vectors
            chosen_vec = doc_vectors[random.randint(0, doc_vectors.shape[0] - 1)]

            self.centroid_vecs.append(chosen_vec)

        # self.position = random.randint(self.MIN_POS, self.MAX_POS)
        # self.velocity = random.randint(self.MIN_VEL, self.MAX_VEL)
        # self.best_pos = self.position
        # self.best_val = np.inf

    def move(self):
        pass

    def calculate_fitness(self):
        pass
