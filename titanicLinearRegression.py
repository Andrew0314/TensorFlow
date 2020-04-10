from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

dftrain = pd.read_csv(TRAIN_DATA_URL)
dftest = pd.read_csv(TEST_DATA_URL)

y_train = dftrain.pop('survived')
y_test = dftest.pop('survived')

categorical = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
numeric = ['age','fare']

feature_columns = []   #data split

for feature in categorical:
    vocab = dftrain[feature].unique()   #gets list of unique values
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature,vocab))

for feature in numeric:
    feature_columns.append(tf.feature_column.numeric_column(feature,dtype=tf.float32))

def make_input_fn(data_df,label_df,num_epochs=20,shuffle=True,batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use
    

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
test_input_fn = make_input_fn(dftest, y_test, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
#result = linear_est.evaluate(test_input_fn)

#print(result["accuracy"])

result = list(linear_est.predict(test_input_fn))
print(dftest.loc[0])   #prints predicted survival perventage
print(y_test.loc[0]) #prints if survived or not
print(result[0]['probabilities'])   #outputs survival chance

