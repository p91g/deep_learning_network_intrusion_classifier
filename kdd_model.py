import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score


#READ DATA
#Read in the entire dataset as a table
X_train = pd.read_csv('C:/Users/patri/Documents/kdd/X_train_cbrt_std_text_bools_pca_clusters.csv')
X_test = pd.read_csv('C:/Users/patri/Documents/kdd/X_test_cbrt_std_text_bools_pca_clusters.csv')

y_train = pd.read_csv('C:/Users/patri/Documents/kdd/Y1_train_dummy.csv')
y_test = pd.read_csv('C:/Users/patri/Documents/kdd/Y1_test_dummy.csv')

class_names = list(y_test.columns)
class_names

# TRAINING DATA %%

# NO PCA OR CLUSTERS # 
no_pca_clusters = np.r_[0:114,129:133]
X_train = X_train.iloc[:,no_pca_clusters]
X_test = X_test.iloc[:,no_pca_clusters]

# convert x_train table to array
X_train_array = X_train.values
X_test_array = X_test.values

X_train_array.shape

# Convert y_train and y_test to arrays
y_train_array = y_train.values
y_test_array = y_test.values

# validation sets - first 10,000 rows for validation
X_valid, X_train_array2 = X_train_array[:10000], X_train_array[10000:]
y_valid, y_train_array2 = y_train_array[:10000], y_train_array[10000:]

# BUILD BASE MODEL # 
#from keras.regularizers import l1_l2

model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[118]),
keras.layers.Dense(100, activation='relu'),
keras.layers.Dense(50, activation='relu'),
keras.layers.Dense(5, activation='softmax')
])

model.compile(loss="categorical_crossentropy",
 optimizer= keras.optimizers.Adam(learning_rate=1e-3),
 metrics=["accuracy"])

# try class weighting for class 4
class_weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0, 4: 2.0}

model.fit(X_train_array, y_train_array, epochs=29, 
                    validation_split=0.1)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# TEST THE MODEL # 
model1_preds_train = model.predict(X_train_array)
model1_test = model.predict(X_test_array)

# Convert one-hot encoded true labels to integer labels
y_train_int = np.argmax(y_train_array, axis=1)
y_test_int = np.argmax(y_test_array, axis=1)

print("MODEL 1")
print("Training Accuracy:", round(accuracy_score(y_train_int, np.argmax(model1_preds_train, axis=1)), 3))
print("Testing Accuracy:", round(accuracy_score(y_test_int, np.argmax(model1_test, axis=1)), 3))

print("Training Precision:", precision_score(y_train_int, np.argmax(model1_preds_train, axis=1), average=None))
print("Testing Precision:", precision_score(y_test_int, np.argmax(model1_test, axis=1), average=None))

print("Training Recall:", recall_score(y_train_int, np.argmax(model1_preds_train, axis=1), average=None))
print("Testing Recall:", recall_score(y_test_int, np.argmax(model1_test, axis=1), average=None))

#accuracy calc w/out dummy = np.mean(np.all(y_train_array == (model1_preds_train == model1_preds_train.max(axis=1, keepdims=True)), axis=1))

# TEST - HYPERPARAMETERS # 
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[118]):
   model = keras.models.Sequential()
   options = {"input_shape": input_shape}
   for layer in range(n_hidden):
    model.add(keras.layers.Dense(n_neurons, activation="relu", **options))
    options = {}
   model.add(keras.layers.Dense(5, activation='softmax'))
   optimizer = keras.optimizers.SGD(learning_rate)
   model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
   return model


keras_classi = keras.wrappers.scikit_learn.KerasClassifier(build_model)

param_distribs = {
    "n_hidden": [0, 1, 2, 3], 
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_classi, param_distribs, n_iter=10, cv=3)

rnd_search_cv.fit(X_train_array2, y_train_array2, epochs=30, validation_data=(X_valid, y_valid), 
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

#Check results
cv_results = rnd_search_cv.cv_results_
fit_times = cv_results['mean_fit_time']
fit_times
'''array([ 88.53302741,  84.57458568,  77.83454863,  80.95840796,
        62.43298197, 134.32457678, 112.12669563, 125.91242353,
       121.13712056, 114.93052363])'''
total_training_time = np.sum(fit_times) * 3
'''3008.294675350189 = 50.1 minutes'''

# best parameters
rnd_search_cv.best_params_
'''{'learning_rate': 0.024082176647226523, 'n_hidden': 3, 'n_neurons': 78}'''

rnd_search_cv.best_score_

# best model
best_model = rnd_search_cv.best_estimator_.model

# test with best model
best_model_preds_train = best_model.predict(X_train_array)
best_model_preds_test = best_model.predict(X_test_array)

# Convert one-hot encoded true labels to integer labels
y_train_int = np.argmax(y_train_array, axis=1)
y_test_int = np.argmax(y_test_array, axis=1)

print("MODEL 1")
print("Training Accuracy:", round(accuracy_score(y_train_int, np.argmax(best_model_preds_train, axis=1)), 3))
print("Testing Accuracy:", round(accuracy_score(y_test_int, np.argmax(best_model_test, axis=1)), 3))

'''Testing Accuracy: 0.998'''

print("Training Precision:", precision_score(y_train_int, np.argmax(model1_preds_train, axis=1), average=None))
print("Testing Precision:", precision_score(y_test_int, np.argmax(model1_test, axis=1), average=None))

print("Training Recall:", recall_score(y_train_int, np.argmax(model1_preds_train, axis=1), average=None))
print("Testing Recall:", recall_score(y_test_int, np.argmax(model1_test, axis=1), average=None))

#save model
best_model.save('best_model.h5')

# load model
from keras.models import load_model
model = load_model('best_model.h5')

