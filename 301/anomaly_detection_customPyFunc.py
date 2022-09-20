# Databricks notebook source
# MAGIC %md
# MAGIC Install Required Packages

# COMMAND ----------

# MAGIC %pip install pyod
# MAGIC %pip install rich

# COMMAND ----------

# MAGIC %md
# MAGIC Import packages

# COMMAND ----------

from pyod.models.deep_svdd import DeepSVDD
import pyod
from pyod.utils.data import evaluate_print
import mlflow
import tempfile
import joblib
import os,shutil
import numpy as np
import pyspark.pandas as ps
import tensorflow.keras as keras
import pandas as pd
import mlflow.pyfunc
import tensorflow as tf
import glob

# COMMAND ----------

# MAGIC %md
# MAGIC Load data from object storage & write data back in delta format

# COMMAND ----------


# File location and type
username='sathish.gangichetty@databricks.com'
# file_location = "/FileStore/tables/creditcard.csv"
input_file_loc = f"/dbfs/Users/{username}/anomaly_detection"
file_type = "delta"
# dbutils.fs.mkdirs(input_file_loc)
# dbutils.fs.cp(file_location, input_file_loc)
# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(input_file_loc)

(df.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema","true")
 .option("path",input_file_loc)
 .saveAsTable('sgfs.cc_data')
)

# COMMAND ----------

# MAGIC %md
# MAGIC Use Pandas on Spark for quick EDA and feature Engineering

# COMMAND ----------

ps.set_option('compute.default_index_type', 'distributed')
input_df = df.pandas_api()
input_df['Class'].value_counts(normalize=True).plot(kind='barh') #just like in pandas

# COMMAND ----------

# MAGIC %md
# MAGIC Some additional prep data so that it becomes ready for an Anomaly detection task. This is an unsupervised learning task, therefore we don't need the `Class` variable. But, we can use the `Class` variable to split the data so we can test the effectiveness of the model

# COMMAND ----------

input_df.reset_index(inplace=True)
input_df.rename(columns={'index':'cust_id'}, inplace=True)
selected_cols = [column for column in input_df.columns if column not in ('Amount')]
sub_df = input_df[selected_cols]
train_df = sub_df[sub_df['Class']==0]
test_anom_df = sub_df[sub_df['Class']==1] #instead of train test split, we're going to split as "good data" vs "not", the assumption is we generally don't know what bad is
train_cols = [column for column in train_df.columns if column not in ('Class')]
train_df = train_df[train_cols]

# COMMAND ----------

# MAGIC %md
# MAGIC - Instantiate an MLFlow Run
# MAGIC - Build Deep SVDD model
# MAGIC - Train the model 
# MAGIC - Log the model
# MAGIC - Log Hyperparameters
# MAGIC - Log Metrics and Save the DeepSVDD model

# COMMAND ----------

with mlflow.start_run(run_name="anomaly_detection") as run:
  # define params for the model
  mlflow.autolog(log_input_examples=True)
  contamination = 0.01
  clf_name = 'DeepSVDD'
  # borrow from the lib
  clf = DeepSVDD(use_ae=False, epochs=1,
                 random_state=123, contamination=.005)
  clf.fit(train_df.to_numpy())
  # log param
  mlflow.log_param("contamination",contamination)
  # get truth data
  y_train = sub_df[sub_df['Class']==0]['Class'].to_numpy()
  # apply model
  y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
  y_train_scores = clf.decision_scores_  # raw outlier scores
  # get test/unseen data & apply model
  X_test = test_anom_df[train_cols].to_numpy()
  y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
  y_test_scores = clf.decision_function(X_test)  # outlier scores
  # log metric to count anomalies "flagged"
  mlflow.log_metric("train_anomalies",y_train_pred.sum()/len(y_train_pred))
  mlflow.log_metric("test_anomalies",y_test_pred.sum()/len(y_test_pred))

  runID = run.info.run_id
  experimentID = run.info.experiment_id
  # The keras model is wrapped by pyod DeepSVDD class, mlflow saves the model - we have to save the state of the class as well to replicate results
  # See here - https://github.com/yzhao062/pyod/issues/328
  # And here - https://github.com/yzhao062/pyod/blob/76959794abde783a486e831265bb3300c1c65b1b/pyod/models/deep_svdd.py#L186
  clf.model_ = None
  temp = tempfile.NamedTemporaryFile(prefix="deepsvdd-", suffix=".joblib")
  temp_name = temp.name
  print(f"{temp.name} is the name")
  try:
    joblib.dump(clf,temp_name)
    mlflow.log_artifact(temp_name, "deepsvdd" )
  finally:
    temp.close()
  
  print(f"MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# Create a folder locally to hold artifacts. Remember that our model has 2 important pieces - the keras model and the wrapper deepSVDD
folder = f'/tmp/artifacts'
if os.path.isdir(folder):
  shutil.rmtree(folder)
  os.makedirs(folder)
  pass
else:
  os.makedirs(folder)

# COMMAND ----------

# MAGIC %md
# MAGIC Reconstruct Model for Batch Scoring

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.download_artifacts(runID, "deepsvdd",folder)
deepSVDD = joblib.load(f"{folder}/deepsvdd/{temp_name.split('/')[-1]}")
logged_model = f'runs:/{runID}/model'
deepSVDD.model_ =  mlflow.keras.load_model(logged_model)
deepSVDD.model_.save(f'{folder}/keras/model.h5')

# COMMAND ----------

!ls -R /tmp/artifacts/

# COMMAND ----------

# MAGIC %md
# MAGIC Run a quick check to see if we yield the same results - check against trained model

# COMMAND ----------

# Sanity Check To See if the model produces the same results - Simple Unit Test
from rich.console import Console
score_array = X_test[0].reshape(1,-1)

def test(test_array:np.array):
  try:
    assert round(deepSVDD.decision_function(test_array)[0],4) == round(y_test_scores[0],4)
    console = Console()
    console.log(locals())
    console.log("Test Passed -- Model Import Runs Ok! ✅")
  except:
    console.log("Test Failed -- Model Reconstruction does not produce the right result ❌ ")
    
test(score_array)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that the model is ready we can safely apply it - locally

# COMMAND ----------

deepSVDD.predict(X_test)[:50] #runs on single machine - has to be dropped into a UDF/python function for parallel scoring

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Package Model for Online Serving - Need a custom pyfunc model to ensure we can properly reconstruct the model

# COMMAND ----------

# capture the env
import platform
import tensorflow
python_version = platform.python_version()
import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      f'python={python_version}',
      'pip',
      {
        'pip': [
          'mlflow',
          'pyod',
          f'tensorflow-cpu=={tensorflow.__version__}',
          f'cloudpickle=={cloudpickle.__version__}',
          f'joblib=={joblib.__version__}',
        ],
      },
    ],
    'name': 'deepsvdd_env'
}

# COMMAND ----------

#link to artifacts - can use paths we saved them on previously
artifacts = {
    "deepSVDD_model": f"{folder}/deepsvdd/{temp_name.split('/')[-1]}",
    "keras_model": f"{folder}/keras/model.h5"
}

# COMMAND ----------

# define a custom class
class PyodDeepSVDDAFWrapper(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    from pyod.models.deep_svdd import DeepSVDD
    import mlflow
    import joblib
    import glob
    import tensorflow as tf
    
    self.deepSVDD_model = joblib.load(glob.glob(context.artifacts["deepSVDD_model"])[0])
    self.deepSVDD_model.model_ =  tf.keras.models.load_model(context.artifacts["keras_model"])
    
  def predict(self,context, model_input):
    logged_model = f'runs:/{runID}/model'
    model_input = model_input.reshape(1,-1)
    return self.deepSVDD_model.predict(model_input)

# COMMAND ----------

# save the model locally
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
mlflow_pyfunc_model_path=f"deepSVDDmodel_pyfunc_"+timestr
print(mlflow_pyfunc_model_path)
mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path,python_model=PyodDeepSVDDAFWrapper(),artifacts=artifacts,
        conda_env=conda_env)

# COMMAND ----------

ls $mlflow_pyfunc_model_path/artifacts/

# COMMAND ----------

# infer model signature
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
loaded_model.predict(score_array)
from mlflow.models.signature import infer_signature
signature = infer_signature(score_array, loaded_model.predict(score_array))
signature

# COMMAND ----------

# log for serving
mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=PyodDeepSVDDAFWrapper(),artifacts=artifacts,
        conda_env=conda_env, signature = signature)
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC To generate a sample for serving

# COMMAND ----------

#get a serving example
import json
json.dumps({"data":score_array.tolist()})
