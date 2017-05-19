import logging
import os
import time

from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from data_helper import DataHelper


class Network:
    __classifiers = {
        'svm': SGDClassifier(),
        'nb': MultinomialNB(alpha=0.05),
        'kmeans': MiniBatchKMeans(n_clusters=25, init='k-means++', n_init=1, init_size=10000, batch_size=1000),
        'perceptron': Perceptron(),
        'passive-aggressive': PassiveAggressiveClassifier()
    }

    def __init__(self, classifier=None, model_file=None):
        if classifier is None and model_file is None:
            raise Exception("Missing parameter")

        if classifier is not None and model_file is not None:
            raise Exception("Too many parameters")

        self.dh = DataHelper()

        if model_file is not None and os.path.exists(model_file):
            self.__load(model_file)
            logging.info("Load model: {0}".format(model_file))
        elif classifier in Network.__classifiers:
            self.clf = Network.__classifiers[classifier]
            logging.info("Set classifier: {0}".format(self.clf.__class__.__name__))
        else:
            raise Exception("Wrong parameter")

    def train(self, parametrize, labels_file, train_dir, test_dir):
        start_time = time.time()

        self.dh.load_labels(labels_file)
        self.dh.parametrize = getattr(DataHelper, 'parametrize_{0}'.format(parametrize))
        self.dh.load_data_dir(train_dir)

        logging.info("Learning...")

        if isinstance(self.clf, MultinomialNB):
            while True:
                x_train, y_train, is_next = self.dh.next_data(1000)
                self.clf.partial_fit(x_train, y_train, range(len(self.dh.label_names)))

                if not is_next:
                    break

            logging.info("Testing...")
            self.dh.load_data_dir(test_dir)
            x_test, y_test, is_next = self.dh.next_data(3000)

            score = metrics.accuracy_score(y_test, self.clf.predict(x_test))
            logging.info("Accuracy: {:.2f}".format(score))

        elif isinstance(self.clf, SGDClassifier):
            while True:
                x_train, y_train, is_next = self.dh.next_data(1000)
                self.clf.fit(x_train, y_train)

                if not is_next:
                    break

            logging.info("Testing...")
            self.dh.load_data_dir(test_dir)
            x_test, y_test, is_next = self.dh.next_data(3000)

            score = metrics.accuracy_score(y_test, self.clf.predict(x_test))
            logging.info("Accuracy: {:.2f}".format(score))

        elif isinstance(self.clf, MiniBatchKMeans):
            while True:
                x_train, y_train, is_next = self.dh.next_data(1000)
                self.clf.partial_fit(x_train)

                if not is_next:
                    break

            labels = self.clf.labels_
            logging.info("Labels: {0}".format(labels))
            self.dh.label_names = dict(zip(labels, labels))

        logging.info("Learning time: {:.2f}s".format(time.time() - start_time))

    def save(self, model_file):
        joblib.dump({'clf': self.clf, 'labels': self.dh.label_names, 'parametrize': self.dh.parametrize.__name__},
                    model_file, compress=True)
        logging.info("Save model: {0}".format(model_file))

    def __load(self, model_file):

        model = joblib.load(model_file)
        self.clf = model['clf']
        self.dh.label_names = model['labels']
        self.dh.parametrize = getattr(DataHelper, model['parametrize'])

    def predict(self, text):
        X = self.dh.text_to_vector(text)
        indexs = self.clf.predict([X])

        return self.dh.label_names.values()[indexs[0]]
