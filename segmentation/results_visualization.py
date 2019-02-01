import colorsys

import numpy as np
import cv2


def plot_TP_FN_FP(input_image, tps, fns, fps, out_file, drawing_params):

    symbols_image = np.zeros_like(input_image)

    for x, y, typ, a in tps:
        drawing_params[typ](symbols_image, x, y, a, (0, 255, 0))
    for x, y, typ, a in fns:
        drawing_params[typ](symbols_image, x, y, a, (0, 0, 255))
    for x, y, typ, a in fps:
        drawing_params[typ](symbols_image, x, y, a, (0, 255, 255))

    res = cv2.addWeighted(input_image, 1, symbols_image, 0.5, 0)
    cv2.imwrite(out_file, res)


def plot_positions(input_image, positions, colors, out_file, drawing_params):

    symbols_image = np.zeros_like(input_image)

    for pos, color in zip(positions, colors):
        for x, y, typ, a in pos:
            drawing_params[typ](symbols_image, x, y, a, color)

    res = cv2.addWeighted(input_image, 1, symbols_image, 0.5, 0)
    cv2.imwrite(out_file, res)


def plot_segmentation_map(input_image, prediction, out_file, num_classes):

    label_im = np.zeros_like(input_image)

    prediction = prediction.astype(np.float32) / (num_classes - 1)

    colors = {k: tuple([int(c * 255) for c in reversed(colorsys.hsv_to_rgb(k, 1, 1))])
              for k in np.unique(prediction)}

    values_x, values_y = np.where(prediction > 0)
    for x, y in zip(values_x, values_y):
        label_im[x, y] = colors[prediction[x, y]]

    res = cv2.addWeighted(input_image, 1, label_im, 0.4, 0)
    cv2.imwrite(out_file, res)
