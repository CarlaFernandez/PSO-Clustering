import os
from os.path import join

class FileManager:
    """This class manages all file operations of the application"""

    def __init__(self, dir_path):
        """Initialize FileManager by loading files into spam and legitimate lists

        Args:
            dir_path: directory in which files are stored
        """
        self.files = self.get_all_from_root(dir_path, '.txt')
        self.texts = []

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

    def load_texts(self, files='all'):
        """
        Loads text from the specified files.
        :param files: which files to get text from
        :return:
        """
        if files == 'all':
            to_load = self.files
        elif files == 'spam':
            to_load = self.spam
        elif files == 'legit':
            to_load = 'legit'
        else:
            print("Unable to read specified files")
            return

        for file in to_load:
            with open(file, 'r') as f:
                self.texts.append(f.read())


