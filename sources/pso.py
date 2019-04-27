import numpy as np

from particle import Particle
import numpy as np

from particle import Particle


class PSO:
    """
    This class performs Particle Swarm Optimization applied to the problem of document clustering.
    """
    def __init__(self, data, file_manager):
        """
        Initialize Particle Swarm Optimization using the provided data.
        :param data: TF-IDF matrix of document vectors.
        """
        print("Initializing PSO...")
        self.data = data
        self.particles = []
        self.file_manager = file_manager

    def clustering(self, iterations=30, num_particles=10, num_clusters=5, distance_metric='cosine', inertia=1, cognitive=2, social=2):
        """
        Perform iterations of the clustering algorithm.
        :param iterations: number of iterations.
        :param num_particles: number of particles.
        :param num_clusters: number of clusters.
        :param distance_metric: distance metric to use.
        :param inertia: inertia value.
        :param cognitive: cognitive factor.
        :param social: social factor.
        :return: document-cluster assignment of the best particle in the swarm.
        """
        if distance_metric != 'euclidean' and distance_metric != 'cosine':
            print("Unknown distance metric '{0}', defaulting to Cosine similarity".format(distance_metric))

        print("Performing clustering...")

        self.__initialize_particles(num_clusters, num_particles)

        self.num_iter = 0
        global_best_fitness = np.inf
        global_best_pos = self.particles[0].centroid_vecs
        global_best_particle = self.particles[0]

        # pca = DocumentPCA()

        # for every iteration until the stop condition is met
        while not self.__stop_condition_met(iterations):

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


            print("Iteration: {0}, best solution: {1}".format(self.num_iter, global_best_fitness))

            self.num_iter += 1


            # pca.visualize_2D_pca(global_best_particle)
            # sleep(0.5)

        # return self.get_spam_stats(global_best_particle)
        return self.get_bbc_stats(global_best_particle)
        # pca.keep_open()


    def get_bbc_stats(self, global_best_particle):
        """
        Prints results for use with the BBC News dataset.
        :param global_best_particle: particle with best document-cluster assignment.
        :return: file with results from the particle.
        """
        output_str = ""
        for key in global_best_particle.assigned:
            print("Cluster {0}".format(key))
            output_str += "Cluster {0}\n".format(key)
            business_counter = 0
            entertainment_counter = 0
            politics_counter = 0
            sport_counter = 0
            tech_counter = 0
            for doc_idx in global_best_particle.assigned[key]:
                doc_name = self.file_manager.files[doc_idx]
                if "business" in doc_name:
                    business_counter += 1
                elif "entertainment" in doc_name:
                    entertainment_counter += 1
                elif "politics" in doc_name:
                    politics_counter += 1
                elif "sport" in doc_name:
                    sport_counter += 1
                elif "tech" in doc_name:
                    tech_counter += 1
            print("---------------------------------------------------------------")
            print("Business: {0}, entertainment: {1}, politics: {2}, sport: {3}, tech: {4}"
                  .format(business_counter, entertainment_counter, politics_counter, sport_counter, tech_counter))
            print("---------------------------------------------------------------")
            output_str += "Business: {0}, entertainment: {1}, politics: {2}, sport: {3}, tech: {4}"\
                .format(business_counter, entertainment_counter, politics_counter, sport_counter, tech_counter)

        return output_str, global_best_particle.fitness

    def __initialize_particles(self, num_clusters, num_particles):
        """
        Initialize the particles that will form the swarm.
        :param num_clusters: number of clusters.
        :param num_particles: nuumber of particles.
        :return: particles created.
        """

        for i in range(num_particles):
            particle = self.__createParticle(self.data, num_clusters)
            self.particles.append(particle)

    def __createParticle(self, doc_vectors, num_clusters):
        """
        Create a single particle.
        :param doc_vectors: document vectors.
        :param num_clusters: number of clusters.
        :return: a single created particle.
        """
        return Particle(doc_vectors, num_clusters)

    def __stop_condition_met(self, iterations):
        """
        Checks if the stop condition for the algorithm has been met.
        :param iterations: number of iterations.
        :return: true if it is met; false otherwise.
        """
        return self.num_iter == iterations

    def get_spam_stats(self, global_best_particle):
        """
        Prints results for use with the Enron Spam dataset.
        :param global_best_particle: particle with best document-cluster assignment.
        :return: file with results from the particle.
        """
        output_str = ""
        for key in global_best_particle.assigned:
            print("Cluster {0}".format(key))
            output_str += "Cluster {0}\n".format(key)
            legit_counter = 0
            spam_counter = 0
            for doc_idx in global_best_particle.assigned[key]:
                doc_name = self.file_manager.files[doc_idx]
                if "legit" in doc_name:
                    legit_counter += 1
                elif "spam" in doc_name:
                    spam_counter += 1
            print("---------------------------------------------------------------")
            print("Legitimate: {0}, spam: {1}".format(legit_counter, spam_counter))
            print("---------------------------------------------------------------")
            output_str+="Legitimate: {0}, spam: {1}".format(legit_counter, spam_counter)

        return output_str, global_best_particle.fitness




