import csv
import os
from os.path import join

import nltk
import numpy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import random


class FileManager:
    """This class manages all file operations of the application"""

    def __init__(self, dir_path):
        """Initialize FileManager by loading files into spam and legitimate lists

        Args:
            dir_path: directory in which files are stored
        """
        self.files = self.__get_all_from_root(dir_path, '.txt')
        self.texts = []

        self.stemmer = PorterStemmer()

        # discard "Summary.txt" files
        self.files = [file for file in self.files if "summary" not in file.lower()]


    def __get_all_from_root(self, folder_path, extension=""):
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
                filename, ext = self.__get_name_extension(joined)
                if ext != "" and ext.lower() == extension:
                    files.append(joined)
        return files

    def __get_name_extension(self, file_path):
        """Returns filename and extension for a file path"""
        return os.path.splitext(file_path)

    def __tokenize(self, text):
        global stop
        tokens = nltk.word_tokenize(text)
        stems = [self.stemmer.stem(token) for token in tokens if token not in stop]
        return stems

    def create_tfidf(self, files='all'):
        """
        Creates the TF-IDF matrix from the specified files.
        :param files: which files to get text from
        :param svd: whether to perform dimensionality reduction by Singular Value Decomposition

        :return: TF-IDF matrix
        """
        if files == 'all':
            to_load = self.files
        elif files == 'testing':
            random.shuffle(self.files)
            print("Loading files...")
            for i in range(len(self.files[:1000])):
                print(i, self.files[i])
            to_load = self.files[:1000]
        else:
            print("Unable to read '{0}' files, defaulting to all files".format(files))
            to_load = self.files

        # using stopwords from nltk
        global stop
        stop = list(stopwords.words('english'))
        print("Creating TF-IDF matrix...")

        vectorizer = TfidfVectorizer(input='filename', tokenizer=self.__tokenize, min_df=2)
        X = vectorizer.fit_transform(to_load)
        print(vectorizer.vocabulary_)

        return X




