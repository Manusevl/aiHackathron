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

# NOT USED (Files loaded from file)
def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)
  return train_file_name, test_file_name
  
def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  author 	= tf.contrib.layers.sparse_column_with_hash_bucket("author", hash_bucket_size=10000)
  language	= tf.contrib.layers.sparse_column_with_hash_bucket("language", hash_bucket_size=100)
  site_url	= tf.contrib.layers.sparse_column_with_hash_bucket("site_url", hash_bucket_size=10000)
  country	= tf.contrib.layers.sparse_column_with_hash_bucket("country", hash_bucket_size=100)
  
  # Continuous base columns.
  year 				= tf.contrib.layers.real_valued_column("year")
  month 			= tf.contrib.layers.real_valued_column("month")
  day 				= tf.contrib.layers.real_valued_column("day")
  hour 				= tf.contrib.layers.real_valued_column("hour")
  minute 			= tf.contrib.layers.real_valued_column("minute")
  txt_words 		= tf.contrib.layers.real_valued_column("txt_words")
  txt_upper 		= tf.contrib.layers.real_valued_column("txt_upper")
  txt_exclamation	= tf.contrib.layers.real_valued_column("txt_exclamation")
  title_words 		= tf.contrib.layers.real_valued_column("title_words")
  title_upper 		= tf.contrib.layers.real_valued_column("title_upper")
  title_exclamation	= tf.contrib.layers.real_valued_column("title_exclamation")

  # Wide columns and deep columns.
  # wide_columns = [gender, native_country, education, occupation, workclass,
                  # relationship, age_buckets,
                  # tf.contrib.layers.crossed_column([education, occupation],
                                                   # hash_bucket_size=int(1e4)),
                  # tf.contrib.layers.crossed_column(
                      # [age_buckets, education, occupation],
                      # hash_bucket_size=int(1e6)),
                  # tf.contrib.layers.crossed_column([native_country, occupation],
                                                   # hash_bucket_size=int(1e4))]
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

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(feature_columns=wide_columns,
										  optimizer=tf.train.FtrlOptimizer(
											learning_rate=0.1,
											l1_regularization_strength=1.0,
											l2_regularization_strength=1.0),
										  model_dir=model_dir)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
  return m

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

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  # train_file_name, test_file_name = maybe_download(train_data, test_data)
  train_file_name = "C:\\Users\\pavanc\\Desktop\\DataSet\\TrainSetFakeNews.csv"
  test_file_name = "C:\\Users\\pavanc\\Desktop\\DataSet\\TestSetFakeNews.csv"
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python",
	  error_bad_lines=False)
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python",
	  error_bad_lines=False)

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  df_train[LABEL_COLUMN] = (df_train["type"].apply(lambda x: "fake" in x)).astype(int)
  df_test[LABEL_COLUMN]  = (df_test["type"].apply(lambda x: "fake" in x)).astype(int)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

FLAGS = None

def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)