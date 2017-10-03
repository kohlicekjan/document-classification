import collections
import glob
import io
import json
import logging
import os

import nltk
import numpy as np

DIC_FILE = '../data/corpus_words_labels.dic'


class DataHelper:
    def __init__(self, dic_file=DIC_FILE):
        self.dic = self.__load_lines(dic_file)
        self.all_words = dict(zip(self.dic, np.zeros(len(self.dic), float)))

        self.label_names = list()
        self.next_num = 0
        self.data_files = list()

    def load_labels(self, file_json):
        with io.open(file_json, 'r', encoding='utf-8')  as f:
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
        with io.open(filename, 'r', encoding='utf-8')  as file_txt:
            # logging.debug("open file: {0}".format(filename))
            lines = file_txt.read().splitlines()

        text = ' '.join(lines).lower()
        x_raw = self.text_to_vector(text)

        label_file = os.path.splitext(os.path.basename(filename))[0].split('_')[1]

        y_raw = self.label_names.keys().index(label_file)

        return x_raw, y_raw

    def text_to_vector(self, text):
        words = nltk.tokenize.RegexpTokenizer(r'[^\d\W]{2,}').tokenize(text)

        words_use = dict(collections.Counter(words))

        all_words = dict(self.all_words)

        all_words.update((k, words_use[k]) for k in set(words_use).intersection(all_words))

        return all_words.values()

    @staticmethod
    def __load_lines(filename):
        with io.open(filename, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())

    @staticmethod
    def create_corpus(data_dir, file_name):
        labels_words = {}
        all_words = {}

        data_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        for filename in data_files:
            with io.open(filename, 'r', encoding='utf-8') as file_txt:
                lines = file_txt.read().splitlines()

            text = ' '.join(lines).lower()
            words = nltk.tokenize.RegexpTokenizer(r'[^\d\W]{2,}').tokenize(text)
            freq = collections.Counter(words)

            label_file = os.path.splitext(os.path.basename(filename))[0].split('_')[1]

            for key_freq in freq.keys():
                if label_file not in labels_words:
                    labels_words[label_file] = {}

                if key_freq not in labels_words[label_file]:
                    labels_words[label_file][key_freq] = 0

                if key_freq not in all_words:
                    all_words[key_freq] = 0

                labels_words[label_file][key_freq] += freq[key_freq]
                all_words[key_freq] += freq[key_freq]

        for key_words in all_words.keys():
            if all_words[key_words] >= 500:
                exists = True
                for key_labels_words in labels_words:
                    if key_words not in labels_words[key_labels_words]:
                        exists = False
                        break

                if exists:
                    del all_words[key_words]

        with open(file_name, 'w') as file_corpus:
            file_corpus.write(u"\n".join(all_words.keys()).encode('utf-8'))
