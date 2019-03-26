# PSO-Clustering

## Core: Done

* Clean data: 
    * Tokenization
    * Stopword removal by Scikit-Learn's English stopword list
    * Stemming by Porter Stemmer
    
* Document representation by Scikit-Learn's TFIDFVectorizer

* Implement objective function by Average Distance of Documents to the Cluster centroid (ADDC)

* Implement PSO algorithm: particles move towards better fitness values 

* Graphical representation of document vectors and centroid vector in space using PCA

## Core: To-Do

* See how to implement cosine correlation metric

## Extensions
* Performance comparison with different parameters
* Implement different algorithms (ACO, K-Means...)
* Supervised classification