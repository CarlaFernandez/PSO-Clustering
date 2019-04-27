import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class DocumentPCA:
    """
    This class allows to visualize a 2D PCA of document vectors and the centroid vectors.
    of a single particle.
    """

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel('Principal Component 1', fontsize=15)
        self.ax.set_ylabel('Principal Component 2', fontsize=15)

    def visualize_2D_pca(self, particle):
        """
        This function shows a visualization of a particle's centroid vectors together with
        the document vectors.
        :param particle: particle whose centroid vectors we will visualize.
        :return: document-centroid plot.
        """

        documents = list(particle.doc_vecs.toarray())

        target = ['Doc' for doc in documents]

        documents.extend(particle.centroid_vecs)
        target.append('Centroid')

        target = pd.DataFrame(data = target, columns = ['target'])

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(documents)

        principalDf = pd.DataFrame(data = principalComponents,
                                   columns = ['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, target], axis=1)

        self.ax.clear()
        targets = ['Doc', 'Centroid']
        colors = ['r', 'g']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['target'] == target
            self.ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=50)
        self.ax.legend(targets)
        self.ax.grid()
        self.fig.canvas.draw()

    def keep_open(self):
        plt.show(block=True)