import argparse

from preprocessing import FileManager
from pso import PSO


def perform_iterations(particles, clusters, iterations, inertia, cognitive, social):
    """
    Performs five runs of the PSO clustering algorithm.
    :param particles: number of particles.
    :param clusters: number of clusters.
    :param iterations: number of iterations.
    :param inertia: inertia parameter.
    :param cognitive: cognitive factor.
    :param social: social factor.
    :return: creates an output file with results from the best run in terms of fitness.
    """
    best_fitness_idx = 0

    all_fitness = []
    all_outputs = []

    for i in range(5):
        print("\n\nSTARTING GLOBAL ITERATION {0}\n\n".format(i + 1))
        pso = PSO(tfidf, file_manager)
        output_str, fitness = pso.clustering(num_particles=particles, num_clusters=clusters, iterations=iterations,
                                             inertia=inertia, cognitive=cognitive, social=social)

        all_fitness.append(fitness)
        all_outputs.append(output_str)

        if fitness < all_fitness[best_fitness_idx]:
            best_fitness_idx = i
    with open("outputs/bbc-{0}-{1}-{2}-{3}-{4}-{5}-{6}".format(iterations, particles, clusters, inertia, cognitive,
                                                                 social, all_fitness[best_fitness_idx]), 'w') as f:
        f.write(all_outputs[best_fitness_idx])


PARTICLES_DEFAULT = 10
INERTIA_DEFAULT = 0.15
COGNITIVE_DEFAULT = 1.5
SOCIAL_DEFAULT = 2.5


parser = argparse.ArgumentParser(usage="Run PSO")
parser.add_argument("-d", "-dir", required=True, help="Folder in which the dataset is stored")
args = parser.parse_args()

# create a FileManager to read the files in -d into a TF-IDF matrix
file_manager = FileManager(args.d)
tfidf = file_manager.create_tfidf(files='all')

# Begin performing the iterations with hyperparameter combinations
particles = PARTICLES_DEFAULT
clusters = 5
iterations = 20
inertia = INERTIA_DEFAULT
cognitive = COGNITIVE_DEFAULT
social = SOCIAL_DEFAULT
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)
inertia = 0.1
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)
inertia = 0.2
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)
inertia = INERTIA_DEFAULT
particles = 5
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)
particles = 15
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)
particles = PARTICLES_DEFAULT
cognitive = 1
social = 3
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)
cognitive = 2
social = 2
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)
cognitive = 2.5
social = 1.5
perform_iterations(particles, clusters, iterations, inertia, cognitive, social)