import argparse
import logging
import os
import shutil
from functools import partial

import numpy as np
import cv2
import tensorflow as tf

from segmentation import model, dataset, bee_dataset
from segmentation.results_analysis import find_positions
from segmentation.results_visualization import plot_positions, plot_segmentation_map

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def predict_data_generator():
    # Here load your 2D image data in the format uint8 (values between 0 and 255)
    # example :
    # for my_image_path in my_images_paths:
    #     yield cv2.imread(my_image_path, cv2.IMREAD_GRAYSCALE)
    raise NotImplementedError()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='checkpoints', help="Path to trained model folder")
    parser.add_argument('--results_folder', default='predict_results', help="Output folder")

    # model parameters, should be the same as training
    parser.add_argument('--num_classes', type=int, default=3, help="How many outputs of the model")
    parser.add_argument('--data_format', type=str, default='channels_last', choices={'channels_last', 'channels_first'})

    # metrics to accept a blob as an object
    parser.add_argument('--min_blob_size_px', type=int, default=20,
                        help="Blobs with bounding box sides smaller than min_blob_size_px are discarded."
                             "Use same coordinate system as the predicted image.")
    parser.add_argument('--max_blob_size_px', type=int, default=200,
                        help="Blobs with bounding box sides larger than max_blob_size_px are discarded."
                             "Use same coordinate system as the predicted image.")

    args = parser.parse_args()
    logger.info('Predicting with settings: {}'.format(vars(args)))

    output_path = os.path.join(os.getcwd(), args.results_folder)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    estimator = tf.estimator.Estimator(model_fn=partial(model.build_model,
                                                        num_classes=args.num_classes,
                                                        data_format=args.data_format,
                                                        bg_fg_weight=None), model_dir=args.checkpoint_dir)

    predictions = estimator.predict(input_fn=partial(dataset.make_dataset,
                                                     data_generator=predict_data_generator,
                                                     data_format=args.data_format,
                                                     batch_size=1,
                                                     mode=tf.estimator.ModeKeys.PREDICT))

    drawing_functions = bee_dataset.get_object_drawing_functions()

    for index, prediction in enumerate(predictions):

        input_image = prediction['input_data']
        pred_image = prediction['prediction']

        channels_axis = 0 if args.data_format == 'channels_first' else -1
        amax = np.argmax(pred_image, axis=channels_axis)

        input_image = np.uint8(np.squeeze(input_image) * 255)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

        plot_segmentation_map(input_image, amax,
                              os.path.join(output_path, "{}_seg_map.png".format(index)), num_classes=args.num_classes)

        predictions_pos = find_positions(amax, args.min_blob_size_px, args.max_blob_size_px)
        if len(predictions_pos) == 0:
            logger.info("Blob analysis failed to find objects.")
            continue

        np.savetxt(os.path.join(output_path, "{}_predictions.csv".format(index)), predictions_pos, fmt="%i,%i,%i,%.4f")

        plot_positions(input_image, [predictions_pos], [(0, 250, 255)],
                       os.path.join(output_path, "{}_positions.png".format(index)),
                       drawing_params=drawing_functions)
