# PSO-Clustering

## Core

* Clean data: 
    * Tokenization.
    * Stopword removal by Scikit-Learn's English stopword list.
    * Stemming by Porter Stemmer.
    
* Document representation by Scikit-Learn's TFIDFVectorizer:
    * A minimum frequency of 2 to include a word in the matrix.

* Cosine correlation distance metric.

* Objective function by Average Distance of Documents to the Cluster centroid (ADDC).

* PSO algorithm: particles move towards better fitness values.


## Extensions
* Performance comparison with different hyperparameters.
* Graphical representation of document vectors and centroid vector in space using PCA.

## Results
* Performance comparison performed on the Enron Spam and BBC News datasets.
* Found in the file results.xlsx.
