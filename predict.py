import argparse
import logging
import os
import shutil
from functools import partial

import numpy as np
import cv2
import tensorflow as tf

from segmentation import model, dataset, bee_dataset, training_config
from segmentation.results_analysis import find_positions, compute_error_metrics
from segmentation.results_visualization import plot_positions, plot_segmentation_map, plot_TP_FN_FP

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def save_error_file(point_d, axis_d, tps, fps, fns, correct_type, out_file):
    total = len(point_d) + fns
    tps_mn = float(tps) / total
    fns_mn = float(fns) / total
    fps_mn = float(fps) / total
    correct_type_pr = float(correct_type) / tps

    with open(out_file, "w") as f:
        f.write("position error (pixels): mean: {:.2f} ({:.2f}) median: {:.2f}\n".format(np.mean(point_d),
                                                                                         np.std(point_d),
                                                                                         np.median(point_d)))
        f.write("correct class: {:.2f}%\n".format(correct_type_pr * 100))
        axis_d = np.rad2deg(axis_d)
        f.write("axis error (degrees) : mean: {:.2f} ({:.2f}) median {:.2f}\n".format(np.mean(axis_d),
                                                                                      np.std(axis_d),
                                                                                      np.median(axis_d)))
        f.write("True Positives: {:.2f}%\n".format(tps_mn * 100))
        f.write("False Negatives: {:.2f}%\n".format(fns_mn * 100))
        f.write("False Positives: {:.2f}%\n".format(fps_mn * 100))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_dir', type=str, help="Path to sample images folder")
    parser.add_argument('--checkpoint_dir', default='checkpoints', help="Path to trained model folder")
    parser.add_argument('--results_folder', default='predict_results', help="Output folder")

    # model parameters, should be the same as training
    parser.add_argument('--num_classes', type=int, default=3, help="How many outputs of the model")
    parser.add_argument('--data_format', type=str, default='channels_last', choices={'channels_last', 'channels_first'})

    # evaluation metrics
    parser.add_argument('--min_distance_px', type=int, default=20,
                        help="Minimum distance in pixels between prediction and label objects"
                             " to be considered a true positive result."
                             "Use same coordinate system as the predicted image.")
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

    images_root_dir = os.path.join(args.dataset_root_dir, 'frames')
    labels_root_dir = os.path.join(args.dataset_root_dir, 'frames_txt')
    if not (os.path.exists(images_root_dir) and os.path.exists(labels_root_dir)):
        raise FileNotFoundError()

    config = training_config.get(args.dataset_root_dir)
    if config is not None:
        to_predict_filenames = config['test']
        logger.info('Predicting only test images from training_config file.')
    else:
        logger.info(
            "Couldn't find a training_config file, so predicting all images in folder {}".format(images_root_dir))
        to_predict_filenames = [os.path.splitext(x)[0] for x in os.listdir(images_root_dir)]

    labels = [bee_dataset.read_label_file_globalcoords(os.path.join(labels_root_dir, name + '.txt'))
              for name in to_predict_filenames]
    regions_of_interest = [l[1] for l in labels]

    estimator = tf.estimator.Estimator(model_fn=partial(model.build_model,
                                                        num_classes=args.num_classes,
                                                        data_format=args.data_format,
                                                        bg_fg_weight=None), model_dir=args.checkpoint_dir)

    predictions = estimator.predict(input_fn=partial(dataset.make_dataset,
                                                     data_generator=partial(bee_dataset.generate_predict,
                                                                            images_root_dir=images_root_dir,
                                                                            filenames=to_predict_filenames,
                                                                            regions_of_interest=regions_of_interest),
                                                     data_format=args.data_format,
                                                     batch_size=1,
                                                     mode=tf.estimator.ModeKeys.PREDICT))

    drawing_functions = bee_dataset.get_object_drawing_functions()

    TP_count, FP_count, FN_count, correct_type_count = 0, 0, 0, 0
    all_pixel_dist, all_axis_diff = [], []

    for name, prediction, label in zip(to_predict_filenames, predictions, labels):
        logger.info('processing {}'.format(name))

        input_image = prediction['input_data']
        pred_image = prediction['prediction']

        channels_axis = 0 if args.data_format == 'channels_first' else -1
        amax = np.argmax(pred_image, axis=channels_axis)

        input_image = np.uint8(np.squeeze(input_image) * 255)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

        plot_segmentation_map(input_image, amax,
                              os.path.join(output_path, "{}_seg_map.png".format(name)), num_classes=args.num_classes)

        predictions_pos = find_positions(amax, args.min_blob_size_px, args.max_blob_size_px)
        if len(predictions_pos) == 0:
            logger.info("Blob analysis failed to find objects.")
            continue

        np.savetxt(os.path.join(output_path, "{}_predictions.csv".format(name)), predictions_pos, fmt="%i,%i,%i,%.4f")
        np.savetxt(os.path.join(output_path, "{}_labels.csv".format(name)), label[0], fmt="%i,%i,%i,%.4f")

        plot_positions(input_image, [label[0], predictions_pos], [(0, 250, 255), (0, 0, 255)],
                       os.path.join(output_path, "{}_mixed.png".format(name)),
                       drawing_params=drawing_functions)

        pixel_dist, axis_diff, correct_type, TP_results, FN_results, FP_results \
            = compute_error_metrics(np.array(label[0]), np.array(predictions_pos), dist_min=args.min_distance_px)

        TP_count += len(TP_results)
        FN_count += len(FN_results)
        FP_count += len(FP_results)
        correct_type_count += correct_type
        all_pixel_dist += pixel_dist
        all_axis_diff += axis_diff

        plot_TP_FN_FP(input_image, TP_results, FN_results, FP_results,
                      os.path.join(output_path, "{}_detail.png".format(name)), drawing_functions)

    save_error_file(all_pixel_dist, all_axis_diff, TP_count, FP_count, FN_count, correct_type_count,
                    os.path.join(output_path, "average_error_metrics.txt"))
