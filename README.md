# PSO-Clustering

## Core: Done

* Clean data: 
    * Tokenization
    * Stopword removal by Scikit-Learn's English stopword list
    * Stemming by Porter Stemmer
    
* Document representation by Scikit-Learn's TFIDFVectorizer

* Implement objective function by Average Distance of Documents to the Cluster centroid (ADDC)

## Core: To-Do
#### Important
* Fix particle movement: they are moving towards higher fitness values

#### Not-so-important
* Different stopping conditions
* Support for local search
* Optimize to provide support for more files (SVD?)

## Extensions
* Performance comparison with different parameters
* Implement different algorithms (ACO, K-Means...)
* Supervised classification