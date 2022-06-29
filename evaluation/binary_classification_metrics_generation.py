import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

from evaluation.utils.data_util import get_confusion_matrix_from_turple
from evaluation.utils.metrics import BinaryClassificationMatrics

csv_file_name = "predic_binary_balanced.csv"

actual_label_index = 1
predicted_label_index = 2
positive_class_value = 1
negative_class_value = 0

BIGGER_SIZE = 20

font = {'family': 'serif',
        'color': '#006633',
        'weight': 'normal',
        'size': BIGGER_SIZE,
        }

matrics = BinaryClassificationMatrics(csv_file_name, actual_label_index, predicted_label_index,
                                      positive_class_value, negative_class_value)
confusion_matrix = matrics.get_confusion_matrix()
matrix = get_confusion_matrix_from_turple(confusion_matrix)
df_cm = pd.DataFrame(matrix, index=[i for i in "PN"],
                     columns=[i for i in "PN"])
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.8)
sn.heatmap(df_cm, annot=True, cmap=plt.get_cmap('BuGn'))
plt.xlabel('Predicted Labels', fontdict=font)
plt.ylabel('Actual Labels', fontdict=font)
plt.xticks(fontsize=BIGGER_SIZE)
plt.yticks(fontsize=BIGGER_SIZE)
plt.show()
print(f"Confusion matrix (tp, fp, tn, fn) : {confusion_matrix}")

accuracy = matrics.get_accuracy()
print(f"Accuracy : {accuracy}")

misclassification_rate = matrics.get_misclassification_rate()
print(f"misclassification_rate : {misclassification_rate}")
precision = matrics.get_precision()
print(f"precision : {precision}")
recall = matrics.get_recall()
print(f"recall : {recall}")
specificity = matrics.get_specificity()
print(f"specificity : {specificity}")
f1_score = matrics.get_f1_score()
print(f"f1_score : {f1_score}")
