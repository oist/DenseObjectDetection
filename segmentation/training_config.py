import json
import logging
import os
import random

import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CONFIG_NAME = 'training_config.json'


def get(dataset_root_dir):
    config_filepath = os.path.join(dataset_root_dir, CONFIG_NAME)

    if not os.path.exists(config_filepath):
        return None

    with open(config_filepath, 'r') as f:
        try:
            training_config = json.load(f)
            logger.info('Loading training config from {}'.format(config_filepath))
            return training_config
        except IOError:
            return None


def create(dataset_root_dir, test_num_files):
    logger.info('Creating new training config')

    all_files = []
    for _, _, files in os.walk(dataset_root_dir):
        for name in files:
            all_files.append(os.path.splitext(name)[0])
    all_files = np.unique(all_files)

    random.shuffle(all_files)
    train_names = all_files[test_num_files:]
    test_names = all_files[:test_num_files]
    with open(os.path.join(dataset_root_dir, CONFIG_NAME), 'w') as outfile:
        json.dump({'test': test_names.tolist(), 'train': train_names.tolist()}, outfile, indent=4)

    return get(dataset_root_dir)
