import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from evaluation.utils.data_util import get_confusion_matrix_from_dict
from evaluation.utils.metrics import MultiClassClassificationConfusionMatrix

actual_label_index = 1
predicted_label_index = 2

no_of_classes = 47
csv_file_name = "predic_type_balanced.csv"
font_size = 35

tick_font_size = 30
font = {'family': 'serif',
        'color': '#006633',
        'weight': 'normal',
        'size': font_size,
        }

title_font = {
    'family': 'serif',
    'color': '#006633',
    'weight': 'normal',
    'size': 45,
}

type_name_list = [
    'Adware',
    'Trojan',
    'Benign',
    'Riskware',
    'Addisplay',
    'Spr',
    'Spyware',
    'Exploit',
    'Downloader',
    'Smssend+Trojan',
    'Troj',
    'Smssend',
    'Clicker+Trojan',
    'Adsware',
    'Malware',
    'Adware+Adware',
    'Rog',
    'Spy',
    'Monitor',
    'Ransom+Trojan',
    'Banker+Trojan',
    'Trj',
    'Gray',
    'Adware+Grayware+Virus',
    'Fakeinst+Trojan',
    'Malware+Trj',
    'Backdoor',
    'Dropper+Trojan',
    'Trojandownloader',
    'Hacktool',
    'Fakeapp',
    'Clickfraud+Riskware',
    'Adload',
    'Addisplay+Adware',
    'Adware+Virus',
    'Clicker',
    'Fakeapp+Trojan',
    'Riskware+Smssend',
    'Rootnik+Trojan',
    'Worm',
    'Fakeangry',
    'Virus',
    'Trojandropper',
    'Adwareare',
    'Risktool+Riskware+Virus',
    'Spy+Trojan',
    'Click'
]
type_name_list.sort()

metrics = MultiClassClassificationConfusionMatrix(csv_file_name, actual_label_index, predicted_label_index)
confusion_matrix, class_summary = metrics.get_confusion_matrix_with_summary()
matrix = get_confusion_matrix_from_dict(confusion_matrix, no_of_classes, class_summary)
indexes = []
for i in range(no_of_classes):
    indexes.append(type_name_list[i])
df_cm = pd.DataFrame(matrix, index=indexes, columns=indexes)
plt.figure(figsize=(64, 50))
sn.set(font_scale=3)
sn.heatmap(df_cm, annot=False, cmap=plt.get_cmap('BuGn'), cbar_kws={"shrink": 0.55})
# plt.title('Confusion Matrix for malware type classification\n', fontdict=title_font)
plt.xlabel('\nPredicted labels', fontdict=font)
plt.ylabel('Actual labels\n\n', fontdict=font)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)

plt.show()
print(f"Confusion matrix (tp, fp, tn, fn) : {confusion_matrix}")
accuracy = metrics.get_accuracy()
print(f"accuracy : {accuracy}")
precision = metrics.get_precision()
average_precision = metrics.get_average_precision()
print(f"precision : {precision}")
recall = metrics.get_recall()
print(f"recall : {recall}")
average_recall = metrics.get_average_recall()
f1_score = metrics.get_f1_score()
print(f"f1_score : {f1_score}")
average_f1_score = metrics.get_average_f1score()
print("average results precision : {0}, recall : {1}, f1-score : {2} ".format(average_precision, average_recall,
                                                                              average_f1_score))
