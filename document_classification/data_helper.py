import collections
import glob
import json
import logging
import numpy as np
import os

import nltk

STOPWORDS_FILE = '../data/stopwords_cz.txt'
DIC_FILE = '../data/corpus_words_cz.dic'


class DataHelper:
    def __init__(self, dic_file=DIC_FILE, stopwords_file=STOPWORDS_FILE):

        self.stopwords = set(self.__load_lines(stopwords_file))

        self.dic = self.__load_lines(dic_file)
        self.all_words = dict(zip(self.dic, np.zeros(len(self.dic), float)))

        self.label_names = list()
        self.next_num = 0
        self.data_files = list()
        self.parametrize = None
        self.data = list()

    def load_labels(self, file_json):
        with open(file_json, 'r') as f:  # encoding='utf-8'
            self.label_names = json.load(f)
            logging.debug("labels: {0}".format(file_json))

    def load_data_dir(self, data_dir):
        self.next_num = 0
        self.data_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))

    def next_data(self, num):
        x_raw = []
        y_raw = []
        i = 0

        for filename in self.data_files[self.next_num:]:
            i += 1

            x, y = self.__load_data_file(filename)
            x_raw.append(x)
            y_raw.append(y)

            if num == i:
                break

        self.next_num += num + 1

        is_next = len(self.data_files) > self.next_num

        return x_raw, y_raw, is_next

    def __load_data_file(self, filename):
        with open(filename, 'r') as file_txt:
            logging.debug("open file: {0}".format(filename))
            lines = file_txt.read().splitlines()

        text = ' '.join(lines).lower()
        x_raw = self.text_to_vector(text)

        label_file = os.path.splitext(os.path.basename(filename))[0].split('_')[1]

        y_raw = self.label_names.keys().index(label_file)

        return x_raw, y_raw

    def text_to_vector(self, text):
        words = nltk.tokenize.RegexpTokenizer(r'\w{3,}').tokenize(text)
        words = self.__remove_stopwords(words)

        words_use = dict(self.parametrize(words))

        all_words = dict(self.all_words)

        all_words.update((k, words_use[k]) for k in set(words_use).intersection(all_words))

        return all_words.values()

    def __remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords and not word.isdigit()]

    @staticmethod
    def parametrize_freq(words):
        return collections.Counter(words)

    @staticmethod
    def parametrize_stop_ten(words):
        freq = collections.Counter(words)
        return {x: y if y <= 10 else 0 for x, y in freq.items()}

    @staticmethod
    def parametrize_exists(words):
        freq = collections.Counter(words)
        return {x: 1 for x, y in freq.items()}

    @staticmethod
    def __load_lines(filename):
        with open(filename, 'r') as f:
            return set(f.read().splitlines())
