import argparse
import logging
import os
from functools import partial

import tensorflow as tf

from segmentation import dataset, bee_dataset, training_config
from segmentation.model import build_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_dir', type=str,
                        help="path of root folder containing frames and frames_txt folders")

    # model parameters
    parser.add_argument('--num_classes', type=int, default=3, help="How many outputs of the model")
    parser.add_argument('--data_format', type=str, default='channels_last', choices={'channels_last', 'channels_first'})

    # training parameters
    parser.add_argument('--bg_fg_weight', type=float, default=0.9,
                        help="How much to weight the foreground objects against the background during training.")
    parser.add_argument('--validation_num_files', type=int, default=10,
                        help="How many images files are used for validation (chosen randomly).")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training")
    parser.add_argument('--num_steps', type=int, default=5000, help="Number of training steps")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Save model to this path.")
    args = parser.parse_args()

    logger.info('Training network with settings: {}'.format(vars(args)))

    images_root_dir = os.path.join(args.dataset_root_dir, 'frames')
    labels_root_dir = os.path.join(args.dataset_root_dir, 'frames_txt')
    if not (os.path.exists(images_root_dir) and os.path.exists(labels_root_dir)):
        raise FileNotFoundError()

    config = training_config.get(args.dataset_root_dir)
    if config is None:
        config = training_config.create(args.dataset_root_dir, args.validation_num_files)

    estimator = tf.estimator.Estimator(model_fn=partial(build_model,
                                                        num_classes=args.num_classes,
                                                        data_format=args.data_format,
                                                        bg_fg_weight=args.bg_fg_weight),
                                       model_dir=args.checkpoint_dir,
                                       config=tf.estimator.RunConfig(save_checkpoints_steps=100,
                                                                     save_summary_steps=100))

    train_spec = tf.estimator.TrainSpec(input_fn=partial(dataset.make_dataset,
                                                         data_generator=partial(bee_dataset.generate_training,
                                                                                frames_root_dir=images_root_dir,
                                                                                labels_root_dir=labels_root_dir,
                                                                                filenames=config['train']),
                                                         data_format=args.data_format,
                                                         batch_size=args.batch_size,
                                                         mode=tf.estimator.ModeKeys.TRAIN), max_steps=args.num_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=partial(dataset.make_dataset,
                                                       data_generator=partial(bee_dataset.generate_training,
                                                                              frames_root_dir=images_root_dir,
                                                                              labels_root_dir=labels_root_dir,
                                                                              filenames=config['test']),
                                                       data_format=args.data_format,
                                                       batch_size=args.batch_size,
                                                       mode=tf.estimator.ModeKeys.EVAL), steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
