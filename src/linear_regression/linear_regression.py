from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

print("Let the Machine Learn!")

train = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
eval = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

print("Getting data...")

dftrain = pd.read_csv(train)
dfeval = pd.read_csv(eval)

# print(dftrain.head())

y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

# print(dftrain.describe())

dftrain.age.hist(bins=20)
# plt.show()

dftrain.sex.value_counts().plot(kind="barh")
# plt.show()

dftrain["class"].value_counts().plot(kind="barh")
# plt.show()

pd.concat([dftrain, y_train], axis=1).groupby("sex").survived.mean().plot(
    kind="barh"
).set_xlabel("% survive")
# plt.show()

CATEGORICAL_COLUMNS = [
    "sex",
    "n_siblings_spouses",
    "parch",
    "class",
    "deck",
    "embark_town",
    "alone",
]
NUMERIC_COLUMNS = ["age", "fare"]

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(
        fc.categorical_column_with_vocabulary_list(feature_name, vocabulary)
    )

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

# print(feature_columns)


def make_input_fn(data_df, label_df, num_epochs=100, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

print("training the model...")
linear_est.train(train_input_fn)
print("evalutaing data...")
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result["accuracy"])

predictions = list(linear_est.predict(eval_input_fn))
for i in range(0, len(dfeval)):
    print(dfeval.loc[i])
    print("Did the person survive? " + str(y_eval.loc[4]))
    print("The model thinks: " + str(predictions[i]["probabilities"][1]))
    print("--------------------------")


print("Done!")
