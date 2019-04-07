import argparse

from preprocessing import FileManager
from pso import PSO

parser = argparse.ArgumentParser(usage="Run PSO")
parser.add_argument("-d", "-dir", required=True, help="Folder in which the dataset is stored")
args = parser.parse_args()

file_manager = FileManager(args.d)
tfidf = file_manager.create_tfidf(files='testing')
pso = PSO(tfidf, file_manager)
pso.clustering(num_particles=10, num_clusters=2, inertia=1.2, cognitive=1.5, social=2.5, distance_metric='euclidean')