import logging
import os
import time

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from data_helper import DataHelper


class Network:
    __NUM_TRAIN_FILE = 1000
    __NUM_TEST_FILE = 3000

    __CLASSIFIERS = {
        'nb': MultinomialNB(alpha=0.05, fit_prior=True),  # 0.85

        'mlpc': MLPClassifier(hidden_layer_sizes=100, activation='identity', solver='adam', alpha=0.0001,
                              batch_size='auto',
                              learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200,
                              shuffle=True, random_state=None, tol=0.0001, warm_start=False,
                              momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                              beta_1=0.9, beta_2=0.999, epsilon=1e-08),  # 0.85

        'svm': SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=5,
                             tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None,
                             learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                             average=False, n_iter=None),  # 0.75, #max_iter = 5, 1000

        'perceptron': Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=5, tol=None, shuffle=True,
                                 verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False,
                                 n_iter=None),  # 0.75, #max_iter = 5, 1000

        'aggressive': PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=5, tol=None, shuffle=True,
                                                  verbose=0, loss='hinge', n_jobs=1, random_state=None,
                                                  warm_start=False, class_weight=None, average=False, n_iter=None),
    # 0.80, #max_iter = 5, 1000

        'bmb': BernoulliNB(alpha=0.05, binarize=0.0, fit_prior=True)  # 0.71
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
        elif classifier in Network.__CLASSIFIERS:
            self.clf = Network.__CLASSIFIERS[classifier]
            logging.info("Set classifier: {0}".format(self.clf.__class__.__name__))
        else:
            raise Exception("Wrong parameter")

    def train(self, labels_file, train_dir, test_dir):
        start_time = time.time()

        self.dh.load_labels(labels_file)
        self.dh.load_data_dir(train_dir)

        logging.info("Learning...")

        while True:
            x_train, y_train, is_next = self.dh.next_data(Network.__NUM_TRAIN_FILE)
            self.clf.partial_fit(x_train, y_train, range(len(self.dh.label_names)))

            if not is_next:
                break

        logging.info("Testing...")
        self.dh.load_data_dir(test_dir)
        x_test, y_test, is_next = self.dh.next_data(Network.__NUM_TEST_FILE)

        score = metrics.accuracy_score(y_test, self.clf.predict(x_test))
        logging.info("Accuracy: {:.2f}".format(score))

        logging.info("Learning time: {:.2f}s".format(time.time() - start_time))

    def save(self, model_file):
        joblib.dump({'clf': self.clf, 'labels': self.dh.label_names}, model_file, compress=True)
        logging.info("Save model: {0}".format(model_file))

    def __load(self, model_file):

        model = joblib.load(model_file)
        self.clf = model['clf']
        self.dh.label_names = model['labels']

    def predict(self, text):
        X = self.dh.text_to_vector(text)
        indexs = self.clf.predict([X])

        return self.dh.label_names.values()[indexs[0]]
