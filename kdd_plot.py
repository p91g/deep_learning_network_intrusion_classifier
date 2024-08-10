import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#READ DATA
#Read in the entire dataset as a table
y_pred = pd.read_csv('C:/Users/patri/Documents/kdd/y_pred.csv', header=None)
y_pred = y_pred.T.values
y_pred

y_test = pd.read_csv('C:/Users/patri/Documents/kdd/Y1_test_dummy.csv')
y_test

class_names = list(y_test.columns)
class_names

y_test_array = y_test.values
y_test_array

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

# Use label_binarize to be multi-label like settings
Y = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
n_classes = Y.shape[1]

# For each class
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y[:, i], y_pred[:, i])
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
    )
    display.plot(ax=plt.gca(), label=class_names[i])

plt.legend()
plt.show()