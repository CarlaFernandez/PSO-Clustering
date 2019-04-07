import argparse

from preprocessing import FileManager
from pso import PSO

parser = argparse.ArgumentParser(usage="Run PSO")
parser.add_argument("-d", "-dir", required=True, help="Folder in which the dataset is stored")
args = parser.parse_args()

file_manager = FileManager(args.d)
tfidf = file_manager.create_tfidf(files='testing')
pso = PSO(tfidf, file_manager)
pso.clustering(num_particles=10, num_clusters=2, iterations=30, inertia=0.10, cognitive=1, social=1, distance_metric='euclidean')