# Databricks notebook source
# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime, date

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as fn
from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

dbutils.widgets.text('select_date', defaultValue=date.today().strftime('%Y-%m-%d'), label="Select Date")
dbutils.widgets.text('input_data_catalog', defaultValue='pl_mlops_demo_prod', label="Input Data Catalog")
dbutils.widgets.text('input_data_schema', defaultValue='mlops_demo', label="Input Data Schema")
dbutils.widgets.text('experiment_name', defaultValue='mlops_demo', label="Experiment Name")
dbutils.widgets.text('model_name', defaultValue='pl_mlops_demo_dev.mlops_demo.propensity', label="Model Name")
dbutils.widgets.text('predictions_table_name', defaultValue='pl_mlops_demo_dev.mlops_demo.predictions', label="Predictions Table Name")
dbutils.widgets.text('commodity', defaultValue='EGGS', label="Commodity")

# COMMAND ----------

select_date = dbutils.widgets.get('select_date')
input_data_catalog = dbutils.widgets.get('input_data_catalog')
input_data_schema = dbutils.widgets.get('input_data_schema')
model_name = dbutils.widgets.get('model_name')
predictions_table_name = dbutils.widgets.get('predictions_table_name')
commodity = dbutils.widgets.get('commodity')

select_date = datetime.strptime(select_date, '%Y-%m-%d').date()
input_data_path = f'{input_data_catalog}.{input_data_schema}'

assert model_name
assert predictions_table_name
assert commodity

# COMMAND ----------

alias = "Champion"
model_uri = f"models:/{model_name}@{alias}"
client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

# COMMAND ----------

to_score = (
  spark.table(f'{input_data_path}.transactions_adj').select('household_key').distinct()
  .withColumn('commodity_desc', fn.lit(commodity))
  .withColumn('select_date', fn.to_date(fn.lit('select_date')))
)

# COMMAND ----------

fe = FeatureEngineeringClient(model_registry_uri='databricks-uc')

ts = datetime.now()

prediction_df = fe.score_batch(
  model_uri=model_uri,
  df=to_score
)

output_df = (
  prediction_df
    .withColumn("prediction", prediction_df["prediction"])
    .withColumn("model_version", fn.lit(model_version))
    .withColumn("inference_timestamp", fn.to_timestamp(fn.lit(ts)))
)

output_df.write.format("delta").mode("overwrite").saveAsTable(predictions_table_name)
