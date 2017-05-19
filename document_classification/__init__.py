"""
Creates models to classify documents into categories.
"""

import os

__author__ = 'Jan Kohlicek'
__version__ = '0.0.1dev'

DEBUG = 1

LOG_FILE = os.getenv('LOGFILE', 'document-classification.log')
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
