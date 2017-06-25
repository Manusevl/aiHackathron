#################################################
# Fake news deep NN classifier using Tensorflow #
#################################################
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import numpy as np
from six.moves import urllib

import pandas as pd
import tensorflow as tf

# Define headers of CSV file
COLUMNS = ["author", "language", "site_url", "country", "year",
           "month", "day", "hour", "minute", "txt_words",
           "txt_upper", "txt_exclamation", "title_words", "title_upper",
           "title_exclamation", "type"]
		   
LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = ["author", "language", "site_url", "country"]
					   
CONTINUOUS_COLUMNS = ["year", "month", "day", "hour", "minute", "txt_words", "txt_upper", 
					  "txt_exclamation", "title_words", "title_upper", "title_exclamation"]
					  
"""Build an estimator."""
# Sparse base columns.
author 		= tf.contrib.layers.sparse_column_with_hash_bucket("author", hash_bucket_size=10000)
language	= tf.contrib.layers.sparse_column_with_hash_bucket("language", hash_bucket_size=100)
site_url	= tf.contrib.layers.sparse_column_with_hash_bucket("site_url", hash_bucket_size=10000)
country		= tf.contrib.layers.sparse_column_with_hash_bucket("country", hash_bucket_size=100)
  
# Continuous base columns.
year 			= tf.contrib.layers.real_valued_column("year")
month 			= tf.contrib.layers.real_valued_column("month")
day 			= tf.contrib.layers.real_valued_column("day")
hour 			= tf.contrib.layers.real_valued_column("hour")
minute 			= tf.contrib.layers.real_valued_column("minute")
txt_words 		= tf.contrib.layers.real_valued_column("txt_words")
txt_upper 		= tf.contrib.layers.real_valued_column("txt_upper")
txt_exclamation	= tf.contrib.layers.real_valued_column("txt_exclamation")
title_words 		= tf.contrib.layers.real_valued_column("title_words")
title_upper 		= tf.contrib.layers.real_valued_column("title_upper")
title_exclamation	= tf.contrib.layers.real_valued_column("title_exclamation")

wide_columns = [author, language, site_url, country, year, month, day, hour, minute, txt_words,
			  txt_upper, txt_exclamation, title_words, title_upper, title_exclamation]
			  
deep_columns = [
  tf.contrib.layers.embedding_column(author, 	dimension=8),
  tf.contrib.layers.embedding_column(language, 	dimension=8),
  tf.contrib.layers.embedding_column(site_url, 	dimension=8),
  tf.contrib.layers.embedding_column(country, 	dimension=8),
  year,
  month,
  day,
  hour,
  minute,
  txt_words,
  txt_upper,
  txt_exclamation,
  title_words,
  title_upper,
  title_exclamation]
		
def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

"""Train and evaluate the model."""
# train_file_name, test_file_name = maybe_download(train_data, test_data)
train_file_name = "C:\\Users\\pavanc\\Desktop\\DataSet\\TrainSetNew.csv"
test_file_name = "C:\\Users\\pavanc\\Desktop\\DataSet\\TestSetNew.csv"

df_train = pd.read_csv(
  tf.gfile.Open(train_file_name),
  names=COLUMNS,
  skipinitialspace=True,
  engine="python",
  encoding="utf_8",
  error_bad_lines=False)
  
df_test = pd.read_csv(
  tf.gfile.Open(test_file_name),
  names=COLUMNS,
  skipinitialspace=True,
  engine="python",
  encoding="utf_8",
  error_bad_lines=False)

# remove NaN elements
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

df_train[LABEL_COLUMN] = (df_train["type"].apply(lambda x: "fake" in x)).astype(int)
df_test[LABEL_COLUMN]  = (df_test["type"].apply(lambda x: "fake" in x)).astype(int)

# Build model
model_dir = "C:\\Users\\pavanc\\Desktop\\DataSet\\Model"
print("model directory = %s" % model_dir)
 
m = tf.contrib.learn.DNNLinearCombinedClassifier(
	model_dir=model_dir,
	linear_feature_columns=wide_columns,
	dnn_feature_columns=deep_columns,
	dnn_hidden_units=[100, 50],
	fix_global_step_increment_bug=True)

# train_steps	= 200
# m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)

results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

for key in sorted(results):
  print("%s: %s" % (key, results[key]))
 
# # Predict  
# predict_file_name = "C:\\Users\\pavanc\\Desktop\\DataSet\\NewSample.csv"
# new_sample =  pd.read_csv(
  # tf.gfile.Open(predict_file_name),
  # names=COLUMNS,
  # skipinitialspace=True,
  # engine="python",
  # encoding="utf_8",
  # error_bad_lines=False)
 
# """Input builder function."""
# # Creates a dictionary mapping from each continuous feature column name (k) to
# # the values of that column stored in a constant Tensor.
# continuous_cols = {k: tf.constant(new_sample[k].values) for k in CONTINUOUS_COLUMNS}
# # Creates a dictionary mapping from each categorical feature column name (k)
# # to the values of that column stored in a tf.SparseTensor.
# categorical_cols = {
  # k: tf.SparseTensor(
	  # indices=[[i, 0] for i in range(new_sample[k].size)],
	  # values=new_sample[k].values,
	  # dense_shape=[new_sample[k].size, 1])
  # for k in CATEGORICAL_COLUMNS}
# # Merges the two dictionaries into one.
# feature_cols = dict(continuous_cols)
# feature_cols.update(categorical_cols)
# # Returns the feature columns and the label.
  
# prediction = list(m.predict(input_fn = feature_cols))

# print(
    # "New Samples, Class Predictions:    {}\n"
    # .format(prediction))