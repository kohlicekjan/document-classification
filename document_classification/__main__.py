import argparse
import logging
import sys
import os

import classifier
from __init__ import __version__, __doc__, LOG_FORMAT, DEBUG
from network import Network


LABELS_FILE = '../data/labels.json'


def valid_dir(parser, arg):
    if not os.path.isdir(arg):
        parser.error("The dir %s does not exist!" % arg)
    else:
        return arg


def parse_args():
    parser = argparse.ArgumentParser(prog='document_classification', description=__doc__)
    parser.add_argument('-v', '--version', action='version', version="document-classification {0}".format(__version__))

    parser.add_argument('--train-dir', dest="train_dir", type=lambda x: valid_dir(parser, x), help="Train data directory (default: %(default)s)")
    parser.add_argument('--test-dir', dest="test_dir",  type=lambda x: valid_dir(parser, x), help="Test data directory (default: %(default)s)")
    parser.add_argument('--parametrize', dest="parametrize",  choices=('exists', 'freq', 'stop_ten'), default='stop_ten', help="Parametrization algorithm (default: %(default)s)")
    parser.add_argument('--classifier', dest="classifier",  choices=('nb', 'svm', 'kmeans'), default='nb', help=" Classification algorithm (default: %(default)s)")
    parser.add_argument('--model-file', dest="model_file", required=True, help="File model for loading or saving", )

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.INFO if not DEBUG else logging.DEBUG
    logging.basicConfig(level=log_level, format=LOG_FORMAT, stream=sys.stdout)

    if os.path.exists(args.model_file):
        nt = Network(model_file=args.model_file)
        classifier.GUI(nt)
    else:
        if args.test_dir is not None and args.test_dir is not None:
            nt = Network(classifier=args.classifier)
            nt.train(args.parametrize, LABELS_FILE, args.train_dir, args.test_dir)
            nt.save(args.model_file)
        else:
            raise Exception("Too many parameters")


if __name__ == "__main__":
    main()
