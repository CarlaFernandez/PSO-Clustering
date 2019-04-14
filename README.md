# PSO-Clustering

## Core

* Clean data: 
    * Tokenization
    * Stopword removal by Scikit-Learn's English stopword list
    * Stemming by Porter Stemmer
    
* Document representation by Scikit-Learn's TFIDFVectorizer

* Cosine correlation and Euclidean distance metrics

* Objective function by Average Distance of Documents to the Cluster centroid (ADDC)

* PSO algorithm: particles move towards better fitness values 

* Graphical representation of document vectors and centroid vector in space using PCA


## Extensions
* Performance comparison with different hyperparameters

## Results
* Performance comparison performed on the Enron Spam and BBC News datasets.
* Found in the file results.xlsx
