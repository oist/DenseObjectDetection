import csv
import logging
import os

import cv2
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# raw image properties
SUB_IMAGE_SIZE = (512, 512)
BEE_OBJECT_SIZES = {1: (20, 35),  # bee class is labeled 1
                    2: (20, 20)}  # butt class is labeled 2
# pre processing params
SCALE_FACTOR = 2  # downscale images, labels by half


def get_object_drawing_functions():
    return {1: draw_bee_body,
            2: draw_bee_butt}


def draw_bee_butt(out_image, x, y, a, color):

    r = 30 // SCALE_FACTOR
    cv2.circle(out_image, (int(x), int(y)), r, color, thickness=2)
    draw_center(out_image, x, y, color)


def draw_center(out_image, x, y, color):

    d = 4 // SCALE_FACTOR
    cv2.rectangle(out_image, (int(x) - d, int(y) - d), (int(x) + d, int(y) + d), color, thickness=-1)


def draw_bee_body(out_image, x, y, a, color):

    d = 60. / SCALE_FACTOR
    dx = np.sin(a) * d
    dy = np.cos(a) * d
    x1, y1, x2, y2 = int(x - dx), int(y + dy), int(x + dx), int(y - dy)
    cv2.line(out_image, (x1, y1), (x2, y2), color, thickness=2)
    draw_center(out_image, x, y, color)


def generate_training(frames_root_dir, labels_root_dir, filenames):

    for name in filenames:

        label_filepath = os.path.join(labels_root_dir, name + '.txt')
        image_filepath = os.path.join(frames_root_dir, name + '.png')

        if not os.path.exists(label_filepath) or not os.path.exists(image_filepath):
            logger.info('Skipping {}.'.format(name))
            continue

        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

        frame_label = read_label_file(label_filepath)

        all_unique_offsets = np.unique([[x[0], x[1]] for x in frame_label], axis=0)

        sub_label_size = (SUB_IMAGE_SIZE[0] // SCALE_FACTOR,
                          SUB_IMAGE_SIZE[1] // SCALE_FACTOR)

        for offset_x, offset_y in all_unique_offsets:

            label_image = np.zeros(sub_label_size, dtype=np.uint8)

            sub_labels = [x for x in frame_label if x[0] == offset_x and x[1] == offset_y]

            for _, _, x, y, bee_type, angle in sub_labels:
                bee_object_size = BEE_OBJECT_SIZES[bee_type]

                x = x // SCALE_FACTOR
                y = y // SCALE_FACTOR
                r1 = bee_object_size[0] // SCALE_FACTOR
                r2 = bee_object_size[1] // SCALE_FACTOR

                ellipse_around_point(label_image, y, x, angle, r1=r1, r2=r2, value=bee_type)

            sub_image = image[offset_y:offset_y + SUB_IMAGE_SIZE[0],
                        offset_x:offset_x + SUB_IMAGE_SIZE[1]]

            fx, fy = (1 / float(SCALE_FACTOR),) * 2
            sub_image = cv2.resize(sub_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

            yield sub_image, label_image


def generate_predict(images_root_dir, filenames, regions_of_interest):

    for name, roi in zip(filenames, regions_of_interest):
        image_filepath = os.path.join(images_root_dir, name + '.png')
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        image = image[roi[2]:roi[3], roi[0]:roi[1]]
        fx, fy = (1 / float(SCALE_FACTOR),) * 2
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        yield image


def ellipse_around_point(image, xc, yc, angle, r1, r2, value):

    image_size = image.shape

    ind0 = np.arange(-xc, image_size[0] - xc)[:, np.newaxis] * np.ones((1, image_size[1]))
    ind1 = np.arange(-yc, image_size[1] - yc)[np.newaxis, :] * np.ones((image_size[0], 1))
    ind = np.concatenate([ind0[np.newaxis], ind1[np.newaxis]], axis=0)

    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    image[((ind[0, :, :] * sin_a + ind[1, :, :] * cos_a) ** 2 / r1 ** 2 + (
            ind[1, :, :] * sin_a - ind[0, :, :] * cos_a) ** 2 / r2 ** 2) <= 1] = value

    return image


def read_label_file(label_filename):
    with open(label_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')

        def parse_row(row):
            offset_x, offset_y = int(row[0]), int(row[1])
            bee_type = int(row[2])
            x, y = int(row[3]), int(row[4])
            angle = float(row[5])

            return offset_x, offset_y, x, y, bee_type, angle

        return list(map(parse_row, csv_reader))


def read_label_file_globalcoords(label_filename):

    rows = read_label_file(label_filename)

    unique_offsets = np.unique([[x[0], x[1]] for x in rows], axis=0)

    roi = [np.min(unique_offsets[:, 0]), np.max(unique_offsets[:, 0]) + SUB_IMAGE_SIZE[0],
           np.min(unique_offsets[:, 1]), np.max(unique_offsets[:, 1]) + SUB_IMAGE_SIZE[1]]

    labels_global_coordinates = [[offset_x + x - roi[0], offset_y + y - roi[2], bee_type, angle]
                                 for offset_x, offset_y, x, y, bee_type, angle in rows]

    labels_global_coordinates = [[x // SCALE_FACTOR, y // SCALE_FACTOR, bee_type, angle]
                                 for x, y, bee_type, angle in labels_global_coordinates]

    return labels_global_coordinates, roi
