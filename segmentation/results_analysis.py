import cv2
import numpy as np


def _find_main_axis(regions, region_index):
    xs, ys = np.where(regions == region_index)
    m = np.concatenate([-ys[:, np.newaxis], xs[:, np.newaxis]], axis=1)

    _, _, v = np.linalg.svd(m - np.mean(m, axis=0), full_matrices=False)

    return np.arctan2(v[0, 0], v[0, 1])


def find_positions(pred, min_blob_size, max_blob_size):
    num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(pred.astype(np.uint8))

    result = []
    for region_index in np.arange(1, num_regions):
        region_stats = stats[region_index]

        # remove too big or too small blobs
        region_width, region_height = region_stats[2], region_stats[3]
        if region_width < min_blob_size or region_width > max_blob_size \
                or region_height < min_blob_size or region_height > max_blob_size:
            continue

        # get blob properties : main axis and centroid
        ax = _find_main_axis(regions, region_index)
        x, y = centroids[region_index][0], centroids[region_index][1]

        # find object type (biggest occurence in the region)
        unique_values, count = np.unique(pred[regions == region_index], return_counts=True)
        typ = unique_values[np.argmax(count)]

        result.append([x, y, typ, ax])

    return result


def _calculate_points_dist(preds, labels):
    # euclidean distance between pred and label
    res = np.zeros((len(preds), len(labels)))

    for ip in range(len(preds)):
        for il in range(len(labels)):
            pred = preds[ip]
            label = labels[il]
            res[ip, il] = np.sqrt((pred[0] - label[0]) ** 2 + (pred[1] - label[1]) ** 2)

    return res


def _axis_difference(a1, a2):
    # transform all angles in range [0, pi/2] before comparing with absolute difference

    a1 = np.mod(a1, np.pi)
    a2 = np.mod(a2, np.pi)

    a1 = np.pi - a1 if a1 > np.pi / 2 else a1
    a2 = np.pi - a2 if a2 > np.pi / 2 else a2

    return np.abs(a2 - a1)


def compute_error_metrics(labels, preds, dist_min):
    TP_results, correct_type, pixel_dist, axis_diff = [], 0, [], []

    dist_matrix = _calculate_points_dist(preds, labels)

    # find matches in dist_matrix, until no more values close enough to match
    while dist_matrix.shape[0] > 0 and dist_matrix.shape[1] > 0 and np.min(dist_matrix) < dist_min:

        ip, il = np.argwhere(dist_matrix == np.min(dist_matrix))[0]

        xp, yp, tp, ap = tuple(preds[ip])
        xl, yl, tl, al = tuple(labels[il])

        TP_results.append(preds[ip])
        pixel_dist.append(dist_matrix[ip, il])

        # only calculate axis difference for class 1 (bee body)
        if tl == 1 and tp == 1:
            axis_diff.append(_axis_difference(ap, al))

        correct_type += int(tp == tl)

        dist_matrix = np.delete(dist_matrix, ip, 0)
        dist_matrix = np.delete(dist_matrix, il, 1)
        preds = np.delete(preds, ip, axis=0)
        labels = np.delete(labels, il, axis=0)

    FN_results = labels
    FP_results = preds

    return pixel_dist, axis_diff, correct_type, TP_results, FN_results, FP_results
