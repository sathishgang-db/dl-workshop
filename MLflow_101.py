# Databricks notebook source
# MAGIC %md
# MAGIC ## MLflow 101 on Databricks
# MAGIC > with Automatic MLflow Logging
# MAGIC 
# MAGIC ### Cluster Requirements:
# MAGIC * ML runtime
# MAGIC 
# MAGIC ### Setup: Imports and Load Data

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

from sklearn import datasets, linear_model, tree
import numpy as np
import pandas as pd
iris = datasets.load_iris()
print("Feature Data: \n", iris.data[::50], "\nTarget Classes: \n", iris.target[::50])

# COMMAND ----------

# MAGIC %md Sidenote: Bamboolib for EDA

# COMMAND ----------

import bamboolib as bam

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1: LogisticRegression

# COMMAND ----------

model_1 = linear_model.LogisticRegression(max_iter=200)
model_1.fit(iris.data, iris.target)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 2: Decision Tree

# COMMAND ----------

model_2 = tree.DecisionTreeClassifier()
model_2.fit(iris.data, iris.target)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable MLflow Autologging and add a context manager

# COMMAND ----------

import mlflow # Import MLflow 
mlflow.autolog(disable=False) # Turn on "autologging"

with mlflow.start_run(run_name="Sklearn Decision Tree"): #Pass in run_name using "with" Python syntax
  model_3 = tree.DecisionTreeClassifier(max_depth=5).fit(iris.data, iris.target) #Instantiate and fit model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Predictions with Model
# MAGIC After registering your model to the model registry and transitioning it to Stage `Production`, load it back to make predictions

# COMMAND ----------

model_name = "sgang_mlflow101" #Or replace with your model name
model_uri = "models:/{}/staging".format(model_name)

print("Loading PRODUCTION model stage with name: '{}'".format(model_uri))
model = mlflow.pyfunc.load_model(model_uri)
print("Model object of type:", type(model))

# COMMAND ----------

predictions = model.predict(pd.DataFrame(iris.data[::50]))
preds = pd.DataFrame(predictions)
preds
