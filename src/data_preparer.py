import re
import csv
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


class DataPreparer:

    #
    def __init__(self, file_path):
        self.documents = []
        self.classes = set()
        self.document_labels = {}
        self.document_count = 0
        self.vocabulary_set = set()
        self.vocabulary_size = 0
        self.indexed_vocabulary = {}

        self.word_document_count = defaultdict(int)
        self.document_word_count = defaultdict(int)

        self.__read_dataset(file_path)  # Read the CSV file and store it on the above variables

    #
    def __read_dataset(self, file_path):
        with open(file_path, 'r') as f:
            for document in csv.reader(f):
                self.classes.add(document[0])
                self.document_labels[self.document_count] = document[0]
                self.documents.append(" ".join(re.split(r'\W+', '{} {}'.format(document[1], document[2]).lower())))
                self.document_count += 1

    #
    def build_vocabulary(self):
        # Count the words in all documents
        for document in self.documents:
            tmp_set = set()
            for word in document.split():
                if word not in tmp_set:
                    self.word_document_count[word] += 1
                    tmp_set.add(word)

        # Find redundant words
        redundant_words_set = set()
        for key, value in self.word_document_count.items():
            if value < 3 or value > self.document_count * 0.4 or key.isdigit():
                redundant_words_set.add(key)

        # Vocabulary properties
        self.vocabulary_set = self.word_document_count.keys() - redundant_words_set
        self.vocabulary_size = len(self.vocabulary_set)

        # Build indexed vocabulary
        vocabulary_dict = {}
        for i, word in enumerate(sorted(list(self.vocabulary_set))):
            vocabulary_dict[word] = i
            self.indexed_vocabulary[i] = word

        return vocabulary_dict

    #
    def generate_document_term_matrix(self):
        vocabulary_dict = self.build_vocabulary()

        sparse_matrix_dict = defaultdict(int)
        for i, document in enumerate(self.documents):
            for word in document.split():
                if word in self.vocabulary_set:
                    sparse_matrix_dict[(i, vocabulary_dict[word])] += 1
                    self.document_word_count[i] += 1

        return sparse_matrix_dict

    #
    def apply_tf_idf(self):
        sparse_matrix_dict = self.generate_document_term_matrix()

        # Calculate TF-IDF weights
        for (document_id, term_id), count in sparse_matrix_dict.items():
            tf = count / self.document_word_count[document_id]
            idf = np.log(self.document_count / self.word_document_count[self.indexed_vocabulary[term_id]])
            sparse_matrix_dict[(document_id, term_id)] = tf * idf

        return self.__dict_to_sparse_matrix(sparse_matrix_dict)

    #
    def __dict_to_sparse_matrix(self, document_term_dict):
        return csr_matrix(
            (list(document_term_dict.values()), zip(*list(document_term_dict.keys()))),
            shape=(self.document_count, self.vocabulary_size)
        )
