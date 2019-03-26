import argparse

from preprocessing import FileManager
from pso import PSO

parser = argparse.ArgumentParser(usage="Run PSO")
parser.add_argument("-d", "-dir", required=True, help="Folder in which the dataset is stored")
args = parser.parse_args()

file_manager = FileManager(args.d)
tfidf = file_manager.create_tfidf(files='testing')
pso = PSO(tfidf)
pso.clustering(num_particles=5, num_clusters=5, inertia=1, cognitive=2, social=2, distance_metric='cosine')