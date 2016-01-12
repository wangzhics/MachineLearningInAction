import numpy as np
from pandas import DataFrame


class VocabNB:
    def __init__(self):
        # train data
        self._train_frame = DataFrame({"vocab": [], "label": []})
        self._vocab_set = set()
        self._vocab_list = list()
        # train result
        self._p0_vec = np.zeros(0)
        self._p1_vec = np.zeros(0)
        self._p0 = 0
        self._p1 = 0

    def set_train_frame(self, train_frame):
        self._train_frame["vocab"] = train_frame["vocab"]
        self._train_frame["label"] = train_frame["label"]

    def add_train_vocabs(self, train_vocabs, train_labels):
        for i in range(len(train_vocabs)):
            self._train_frame = \
                self._train_frame.append({"vocab": train_vocabs[i], "label": train_labels[i]}, ignore_index=True)

    def train_nb(self):
        for vocab_row in self._train_frame["vocab"]:
            self._vocab_set = self._vocab_set | set(vocab_row)
        self._vocab_list = list(self._vocab_set)
        # build the train matrix
        matrix_row_size = len(self._train_frame)
        matrix_col_size = len(self._vocab_set)
        train_matrix = np.zeros((matrix_row_size, matrix_col_size))
        # calculate the probability
        i = 0
        for vocab_row in self._train_frame["vocab"]:
            j = 0
            for vocab in self._vocab_list:
                if vocab in vocab_row:
                    train_matrix[i][j] = 1
                else:
                    train_matrix[i][j] = 0
                j += 1
            i += 1
        p1 = sum(self._train_frame["label"]) / float(matrix_row_size)
        p0 = 1 - p1
        p0_count_vec = np.ones(matrix_col_size)
        p1_count_vec = np.ones(matrix_col_size)
        p0_count = matrix_col_size
        p1_count = matrix_col_size
        matrix_labels = self._train_frame["label"]
        for i in range(matrix_row_size):
            if matrix_labels.iloc[i] == 1:
                p1_count_vec += train_matrix[i]
                p1_count += sum(train_matrix[i])
            else:
                p0_count_vec += train_matrix[i]
                p0_count += sum(train_matrix[i])
        # the final naive bayes probability vector
        self._p0_vec = np.log(p0_count_vec / p0_count)
        self._p1_vec = np.log(p1_count_vec / p1_count)
        self._p0 = np.log(p0)
        self._p1 = np.log(p1)

    def classify(self, classify_vocabs):
        classify_vec = np.zeros(len(self._vocab_set))
        i = 0
        for vocab in self._vocab_list:
            if vocab in classify_vocabs:
                classify_vec[i] = 1
            else:
                classify_vec[i] = 0
            i += 1
        p1 = self._p1 + sum(classify_vec * self._p1_vec)
        p0 = self._p0 + sum(classify_vec * self._p0_vec)
        if p1 > p0:
            return 1
        return 0