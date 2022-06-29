from sklearn.metrics import accuracy_score

from evaluation.utils.csv_utils import CSVUtils


class BinaryClassificationMatrics:
    def __init__(self, results_csv, actual_label_index, predicted_label_index, positive_class_value,
                 negative_class_value):
        self.data_rows = CSVUtils.read_csv(results_csv)
        self.actual_label_index = actual_label_index
        self.predicted_label_index = predicted_label_index
        self.positive_class_value = str(positive_class_value)
        self.negative_class_value = str(negative_class_value)

    def get_confusion_matrix(self):
        tp, tn, fp, fn = 0, 0, 0, 0
        for row in self.data_rows:
            actual_value = row[self.actual_label_index]
            predicted_value = row[self.predicted_label_index]

            if actual_value == self.positive_class_value:
                if predicted_value == actual_value:
                    tp += 1
                else:
                    fn += 1
            elif actual_value == self.negative_class_value:
                if predicted_value == actual_value:
                    tn += 1
                else:
                    fp += 1
            else:
                print(f"Error in data type. value datatype : {type(self.positive_class_value)} "
                      f"and actual value datatype : {type(actual_value)}")

        return tp, fp, tn, fn

    def get_accuracy(self):
        tp, fp, tn, fn = self.get_confusion_matrix()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    def get_misclassification_rate(self):
        tp, fp, tn, fn = self.get_confusion_matrix()
        misclassification_rate = (fp + fn) / (tp + tn + fp + fn)
        return misclassification_rate

    def get_precision(self):
        tp, fp, tn, fn = self.get_confusion_matrix()
        precision = tp / (tp + fp)
        return precision

    def get_recall(self):
        tp, fp, tn, fn = self.get_confusion_matrix()
        recall = tp / (tp + fn)
        return recall

    def get_specificity(self):
        tp, fp, tn, fn = self.get_confusion_matrix()
        specificity = tn / (tn + fp)

        return specificity

    def get_f1_score(self):
        tp, fp, tn, fn = self.get_confusion_matrix()
        f1_score = (2 * tp) / (2 * tp + fp + fn)

        return f1_score


class MultiClassClassificationConfusionMatrix:
    def __init__(self, results_csv, actual_label_index, predicted_label_index):
        self.data_rows = CSVUtils.read_csv(results_csv)
        self.actual_label_index = actual_label_index
        self.predicted_label_index = predicted_label_index

    def get_confusion_matrix(self):
        confusion_matrix = {}

        for row in self.data_rows:
            actual_value = row[self.actual_label_index]
            predicted_value = row[self.predicted_label_index]

            actual_value_dict = confusion_matrix.get(actual_value, {})
            predicted_value_matrix = actual_value_dict.get(predicted_value, 0)
            predicted_value_matrix += 1
            actual_value_dict[predicted_value] = predicted_value_matrix
            confusion_matrix[actual_value] = actual_value_dict
        return confusion_matrix

    def get_confusion_matrix_with_summary(self):
        confusion_matrix = {}
        summary_values_dict = {}
        for row in self.data_rows:
            actual_value = row[self.actual_label_index]
            predicted_value = row[self.predicted_label_index]

            actual_value_dict = confusion_matrix.get(actual_value, {})
            predicted_value_matrix = actual_value_dict.get(predicted_value, 0)
            predicted_value_matrix += 1
            actual_value_dict[predicted_value] = predicted_value_matrix
            confusion_matrix[actual_value] = actual_value_dict

            summary_values_dict[actual_value] = summary_values_dict.get(actual_value, 0) + 1

        return confusion_matrix, summary_values_dict

    def get_accuracy(self):
        actual_labels = []
        predicted_labels = []
        for row in self.data_rows:
            actual_value = row[self.actual_label_index]
            predicted_value = row[self.predicted_label_index]

            actual_labels.append(actual_value)
            predicted_labels.append(predicted_value)

        accuracy = accuracy_score(actual_labels, predicted_labels)
        return accuracy

    def get_matrics(self):
        confusion_matrix = self.get_confusion_matrix()
        classes = confusion_matrix.keys()
        matrics = {}
        for class_name in classes:
            other_classes = list(classes)
            other_classes.remove(class_name)

            tp = confusion_matrix[class_name].get(class_name, 0)
            tn = 0
            for other_class_r in other_classes:
                for other_class_c in other_classes:
                    row = confusion_matrix.get(other_class_r, {})
                    value = row.get(other_class_c, 0)
                    tn += value
            fn = 0
            for other_class in other_classes:
                value = confusion_matrix[class_name].get(other_class, 0)
                fn += value

            fp = 0
            for other_class in other_classes:
                value = confusion_matrix.get(other_class, {}).get(class_name, 0)
                fp += value

            matrics[class_name] = (tp, tn, fp, fn)
        return matrics

    @staticmethod
    def __get_average_from_dict(dictionary):
        keys = dictionary.keys()
        total = 0
        for key in keys:
            value = dictionary.get(key, 0)
            if value == "NA":
                value = 0
            elif isinstance(value, str):
                value = 0
            total += value
        average = total / len(keys)

        return average

    def get_precision(self):
        matrics = self.get_matrics()
        precision_result = {}

        for class_name in matrics.keys():
            (tp, tn, fp, fn) = matrics.get(class_name, (0, 0, 0, 0))
            if tp + fp == 0:
                precision = "NA"
            else:
                precision = tp / (tp + fp)
            precision_result[class_name] = precision

        return precision_result

    def get_average_precision(self):
        precision_result = self.get_precision()
        average = self.__get_average_from_dict(precision_result)

        return average

    def get_recall(self):
        matrics = self.get_matrics()
        recall_result = {}

        for class_name in matrics.keys():
            (tp, tn, fp, fn) = matrics.get(class_name, (0, 0, 0, 0))
            if tp + fn == 0:
                recall = "NA"
            else:
                recall = tp / (tp + fn)
            recall_result[class_name] = recall

        return recall_result

    def get_average_recall(self):
        recall_results = self.get_recall()
        average = self.__get_average_from_dict(recall_results)

        return average

    def get_f1_score(self):
        matrics = self.get_matrics()
        f1_score_result = {}

        for class_name in matrics.keys():
            (tp, tn, fp, fn) = matrics.get(class_name, (0, 0, 0, 0))
            if tp + fp + fn == 0:
                f1_score = "NA"
            else:
                f1_score = (2 * tp) / (2 * tp + fp + fn)
            f1_score_result[class_name] = f1_score

        return f1_score_result

    def get_average_f1score(self):
        f1_score_results = self.get_f1_score()
        average = self.__get_average_from_dict(f1_score_results)

        return average
