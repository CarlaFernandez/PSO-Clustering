import argparse

from preprocessing import FileManager
from pso import PSO

parser = argparse.ArgumentParser(usage="Run PSO")
parser.add_argument("-d", "-dir", required=True, help="Folder in which the dataset is stored")
args = parser.parse_args()

file_manager = FileManager(args.d)
tfidf = file_manager.create_tfidf(files='testing')
pso = PSO(tfidf)
pso.clustering(num_particles=2, num_clusters=5, distance_metric='euclidean')