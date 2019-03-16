import os
from os.path import join

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from nltk.stem.porter import PorterStemmer

class FileManager:
    """This class manages all file operations of the application"""

    def __init__(self, dir_path):
        """Initialize FileManager by loading files into spam and legitimate lists

        Args:
            dir_path: directory in which files are stored
        """
        self.files = self.get_all_from_root(dir_path, '.txt')
        self.texts = []

        self.stemmer = PorterStemmer()

        # discard "Summary.txt" files
        self.files = [file for file in self.files if "summary" not in file.lower()]

        # divide into spam and legit
        self.spam = [file for file in self.files if "spam" in file.lower()]
        self.legit = [file for file in self.files if "ham" in file.lower()]

        print("Loaded {0} spam files, {1} legitimate files".format(len(self.spam), len(self.legit)))

    def get_all_from_root(self, folder_path, extension=""):
        """Gets files from dirs and subdirs starting from a root path.

        Args:
            folder_path: root path to begin the search
            extension: extension of files to search
        Returns:
            file paths of all files under folder_path"""

        files = []
        for r, d, f in os.walk(folder_path):
            for file in f:
                joined = join(r, file)
                filename, ext = self.get_name_extension(joined)
                if ext != "" and ext.lower() == extension:
                    files.append(joined)
        return files

    def get_name_extension(self, file_path):
        """Returns filename and extension for a file path"""
        return os.path.splitext(file_path)

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        return stems

    def create_tfidf(self, files='all', svd=False):
        """
        Creates the TF-IDF matrix from the specified files.
        :param files: which files to get text from
        :param svd: whether to perform dimensionality reduction by Singular Value Decomposition

        :return: TF-IDF matrix
        """
        if files == 'all':
            to_load = self.files
        elif files == 'spam':
            to_load = self.spam
        elif files == 'legit':
            to_load = 'legit'
        # TODO remove this entry
        elif files == 'testing':
            to_load = self.files[:50]
        else:
            print("Unable to read '{0}' files, defaulting to all files".format(files))
            to_load = self.files

        # using stopwords from nltk
        stop = list(stopwords.words('english'))

        print("Creating TF-IDF matrix...")
        text = ""
        for file in to_load:
            with open(file, 'r') as f:
                text += f.read()
        vectorizer = TfidfVectorizer(text, tokenizer=self.tokenize)
        X = vectorizer.fit_transform(to_load)
        if svd:
            print("Performing SVD...")
            X = TruncatedSVD().fit_transform(X)
        return X





