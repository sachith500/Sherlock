import numpy as np


def get_confusion_matrix_from_dict(data_dict: dict, no_of_classes, class_summary):
    confusion_matrix = np.zeros([no_of_classes, no_of_classes], dtype=float)
    for i in data_dict.keys():
        class_results = data_dict.get(i, {})
        for j in class_results.keys():
            value = class_results.get(j, 0)
            confusion_matrix[int(i)][int(j)] = float(value / class_summary[i])

    for i in range(no_of_classes):
        v = class_summary.get(str(i))
        if v is None:
            print(i)

    return confusion_matrix


def get_confusion_matrix_from_turple(data_turple):
    (tp, fp, tn, fn) = data_turple
    confusion_matrix = np.zeros([2, 2], dtype=float)
    confusion_matrix[0][0] = round(tp / (tp + fp), 2)
    confusion_matrix[0][1] = round(fp / (tp + fp), 2)
    confusion_matrix[1][0] = round(fn / (tn + fn), 2)
    confusion_matrix[1][1] = round(tn / (tn + fn), 2)

    return confusion_matrix


def print_confifusion_matrix_values(confusion_matrix):
    pass
