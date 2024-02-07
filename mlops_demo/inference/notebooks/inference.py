# Databricks notebook source
# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime, date

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as fn

import databricks.feature_store as feature_store
from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

dbutils.widgets.text('select_date', defaultValue=date.today().strftime('%Y-%m-%d'), label="Select Date")
dbutils.widgets.text('input_data_catalog', defaultValue='pl_mlops_demo_prod', label="Input Data Catalog")
dbutils.widgets.text('input_data_schema', defaultValue='mlops_demo', label="Input Data Schema")
dbutils.widgets.text('features_catalog', defaultValue='pl_mlops_demo_dev', label="Features Catalog")
dbutils.widgets.text('features_schema', defaultValue='mlops_demo', label="Features Schema")
dbutils.widgets.text('model_name', defaultValue='pl_mlops_demo_dev.mlops_demo.propensity', label="Model Name")
dbutils.widgets.text('predictions_table_name', defaultValue='pl_mlops_demo_dev.mlops_demo.predictions', label="Predictions Table Name")
dbutils.widgets.text('commodity', defaultValue='EGGS', label="Commodity")

# COMMAND ----------

select_date = dbutils.widgets.get('select_date')
input_data_catalog = dbutils.widgets.get('input_data_catalog')
input_data_schema = dbutils.widgets.get('input_data_schema')
features_catalog = dbutils.widgets.get('features_catalog')
features_schema = dbutils.widgets.get('features_schema')
model_name = dbutils.widgets.get('model_name')
predictions_table_name = dbutils.widgets.get('predictions_table_name')
commodity = dbutils.widgets.get('commodity')

select_date = datetime.strptime(select_date, '%Y-%m-%d').date()
input_data_path = f'{input_data_catalog}.{input_data_schema}'
features_path = f'{features_catalog}.{features_schema}'

assert model_name
assert predictions_table_name
assert commodity

# COMMAND ----------

raw_data = (
  spark.table(f'{input_data_path}.transactions_adj').select('household_key').distinct()
  .withColumn('commodity_desc', fn.lit(commodity))
  .withColumn('select_date', fn.to_date(fn.lit(select_date)))
)

# COMMAND ----------

feature_lookups = [
  # household features
  feature_store.FeatureLookup(
    table_name = f'{features_path}.household_features',
    lookup_key = ['household_key','select_date'],
    feature_names = [c for c in spark.table(f'{features_path}.household_features').drop('household_key','select_date').columns],
    rename_outputs = {c:f'household__{c}' for c in spark.table(f'{features_path}.household_features').columns}
    ),
  # commodity features
  feature_store.FeatureLookup(
    table_name = f'{features_path}.commodity_features',
    lookup_key = ['commodity_desc','select_date'],
    feature_names = [c for c in spark.table(f'{features_path}.commodity_features').drop('commodity_desc','select_date').columns],
    rename_outputs = {c:f'commodity__{c}' for c in spark.table(f'{features_path}.commodity_features').columns}
    ),
  # household-commodity features
  feature_store.FeatureLookup(
    table_name = f'{features_path}.household_commodity_features',
    lookup_key = ['household_key','commodity_desc','select_date'],
    feature_names = [c for c in spark.table(f'{features_path}.household_commodity_features').drop('household_key','commodity_desc','select_date').columns],
    rename_outputs = {c:f'household_commodity__{c}' for c in spark.table(f'{features_path}.household_commodity_features').columns}
    )
  ]

# COMMAND ----------

fs = FeatureStoreClient()

to_score = fs.create_training_set(
  df=raw_data,
  feature_lookups=feature_lookups,
  label=''
).load_df()

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient(registry_uri="databricks-uc")

alias = "Champion"
model_uri = f"models:/{model_name}@{alias}"

model_version = client.get_model_version_by_alias(model_name, alias).version
model_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

output_df = (
  to_score
    .withColumn('prediction', model_udf())
    .select('household_key','commodity_desc','select_date','prediction')
    .withColumn("model_version", fn.lit(model_version))
    .withColumn("inference_timestamp", fn.to_timestamp(fn.lit(datetime.now())))
)

output_df.write.format("delta").mode("overwrite").saveAsTable(predictions_table_name)
