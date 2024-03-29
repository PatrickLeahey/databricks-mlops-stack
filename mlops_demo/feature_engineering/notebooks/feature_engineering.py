# Databricks notebook source
# MAGIC %md The purpose of this notebook is to engineer the features required for our propensity scoring work. This notebook was developed using the **Databricks 13.3 LTS ML** runtime.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC In this notebook we will leverage transactional data associated with individual households to generate features with which we will train our model and, later, perform inference, *i.e.* make predictions.  Our goal is to predict the likelihood a household will purchase products from a given product category, *i.e.* commodity-designation, in the next 30 days. 
# MAGIC
# MAGIC In an operational workflow which we can imagine running separately from this, we would receive new data into the lakehouse on a periodic, *i.e.* daily or more frequent, basis.  As that data arrives, we might recalculate new or updated features and store these for the purpose of making predictions about a future period. To train a model, we'd need the state of these features some period of time, *i.e.* 30 days in this scenario, behind the current features. For this reason, it will be important to keep past versions of features for some limited duration, *i.e.* at least 30 days in this scenario.
# MAGIC
# MAGIC This notebook represents the logic associated with training one set of features for a single date. The *current_date* will be calculated in a separate notebooks and either accessed as part of the workflow or passed directly into this notebook via a widget.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import os
from importlib import import_module

from pyspark.sql.types import *
import pyspark.sql.functions as fn
from pyspark.sql.window import Window

from databricks.feature_store import FeatureStoreClient

from datetime import datetime, timedelta, date

# COMMAND ----------

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../modules

mod = import_module("get_features")
get_features = getattr(mod, "get_features")

# COMMAND ----------

# MAGIC %md ##Step 1: Access Data from which to Derive Features
# MAGIC
# MAGIC This notebook will be typically run as part of a regularly scheduled workflow.  However, during development and initialization of the feature store, we should expect to see it run manually.  To support either scenario, we will define [widgets](https://docs.databricks.com/notebooks/widgets.html) through which values can be assigned to the notebook, either directly or through a runtime call from another notebook.  We will first attempt to retrieve configuration values from the jobs workflow but failing that, we will fallback to values supplied by these widgets:

# COMMAND ----------

# DBTITLE 1,Create Widgets
dbutils.widgets.text('select_date', defaultValue=date.today().strftime('%Y-%m-%d'), label="Select Date")
dbutils.widgets.text('input_data_catalog', defaultValue='pl_mlops_demo_prod', label="Input Data Catalog")
dbutils.widgets.text('input_data_schema', defaultValue='mlops_demo', label="Input Data Schema")
dbutils.widgets.text('features_catalog', defaultValue='pl_mlops_demo_dev', label="Features Catalog")
dbutils.widgets.text('features_schema', defaultValue='mlops_demo', label="Features Schema")

# COMMAND ----------

# DBTITLE 1,Get Widget Values
select_date = '2019-10-01' # select_date = dbutils.widgets.get('select_date')
input_data_catalog = dbutils.widgets.get('input_data_catalog')
input_data_schema = dbutils.widgets.get('input_data_schema')
features_catalog = dbutils.widgets.get('features_catalog')
features_schema = dbutils.widgets.get('features_schema')

select_date = datetime.strptime(select_date, '%Y-%m-%d').date()
input_data_path = f'{input_data_catalog}.{input_data_schema}'
features_path = f'{features_catalog}.{features_schema}'

# COMMAND ----------

# MAGIC %md Using our configuration values, we can now retrieve the transaction data from which we will derive our features:

# COMMAND ----------

# DBTITLE 1,Get Transaction Inputs Up to Current Date
transactions = (
    spark
      .table(f'{input_data_path}.transactions_adj')
      .join(
        spark.table(f'{input_data_path}.products'), # join to products to get commodity assignment
        on='product_id',
        how='inner'
        )
      .filter(fn.expr(f"day <= '{select_date}'"))
    )

# COMMAND ----------

# MAGIC %md In addition to the raw transaction data, we need to assemble the full set of all households and commodities for which we may wish to derive features:

# COMMAND ----------

# DBTITLE 1,Get All Household-Commodity Combinations Possible for this Period
# get unique commodities

commodities_to_score = (
  spark
    .table(f'{input_data_path}.transactions_adj')
    .join(spark.table(f'{input_data_path}.products'), on='product_id')
    .select('commodity_desc','basket_id')
    .groupBy('commodity_desc')
      .agg(fn.countDistinct('basket_id').alias('purchases'))
    .orderBy('purchases', ascending=False)
    .limit(10)
    .select(fn.expr("regexp_replace(commodity_desc, '[-|\\/:;,.\"'']', '_')").alias('commodity_desc'))
    .withColumn('commodity_desc', fn.expr("replace(commodity_desc, ' ', '_')"))
)

# get unique households
households = transactions.select('household_key').distinct()

# cross join all commodities and households
household_commodity = households.crossJoin(commodities_to_score)

# COMMAND ----------

# MAGIC %md ## Step 2: Define Feature Generation Logic
# MAGIC
# MAGIC The feature generation logic will be used to derive values from 30 day, 60 day, 90 day and 1 year prior to the *select_date*. A wide range of metrics will be calculated for each period, but what is captured here is by no means exhaustive of what we could derive from this dataset.  The encapsulation of this logic as a function will allow us to re-use this logic as we derive features for households, commodities and household-commodity combinations later:

# COMMAND ----------

# MAGIC %md ##Step 3: Generate Household Features
# MAGIC
# MAGIC Using our transaction inputs, we can derive household-level features as follows:

# COMMAND ----------

# DBTITLE 1,Calculate Household Features
# features will be grouped on households
grouping_keys = ['household_key']

# get master set of household keys in incoming data
features = (
  household_commodity
    .select(grouping_keys)
    .distinct()
    .withColumn('select_date', fn.lit(select_date))
    )

# calculate household features for each period and join to master set
for window in ['30d','60d','90d','1yr']:
  features = (
    features
      .join(
          get_features(df=transactions, grouping_keys=grouping_keys, window=window), 
          on=grouping_keys, 
          how='leftouter'
        )
    )

# fill-in any missing values
household_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_keys])

# COMMAND ----------

# MAGIC %md We can now write these data to our feature store as follows.  Please note that we are using the *household_key* field in combination with the *day* field for the unique identifier for these records.  While feature store tables support a timestamp column for versioning of records (as part of the [time series feature table](https://docs.databricks.com/machine-learning/feature-store/time-series.html) capability), in practice we have found use of this feature to be very slow compared to just placing the timestamp in the primary key.  The key - no pun intended - to making this hack work is that you must have a perfect match for the timestamp value in data used to retrieve features.  The time series feature table capability allows you to retrieve the feature version available at a given point in time but does not require a perfect match:

# COMMAND ----------

# DBTITLE 1,Write Features to Feature Store
# instantiate feature store client
fs = FeatureStoreClient()

# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f'{features_path}.household_features')
except: # create it now
  pass
  _ = (
    fs
      .create_table(
        name=f'{features_path}.household_features', # name of feature store table
        primary_keys= grouping_keys + ['select_date'], # name of keys that will be used to locate records
        schema=household_features.schema, # schema of feature set as derived from our feature_set dataframe
        description='household features used for propensity scoring' 
      )
    )

# merge feature set data into feature store
_ = (
  fs
    .write_table(
      name=f'{features_path}.household_features',
      df = household_features,
      mode = 'merge' # merge data into existing feature set, instead of 'overwrite'
    )
  )

# COMMAND ----------

# MAGIC %md We can verify our data by retrieving features from the feature table for the *select_date*:

# COMMAND ----------

# MAGIC %md ##Step 4: Generate Commodity Features
# MAGIC
# MAGIC We can now do the same for each commodity in the dataset:

# COMMAND ----------

# DBTITLE 1,Calculate Commodity Features
# features will be grouped on households
grouping_keys = ['commodity_desc']

# get master set of household keys in incoming data
features = (
  household_commodity
    .select(grouping_keys)
    .distinct()
    .withColumn('select_date', fn.lit(select_date)) # assign date to feature set
    )

# calculate household features for each period and join to master set
for window in ['30d','60d','90d','1yr']:
  features = (
    features
      .join(
          get_features(transactions, grouping_keys, window=window), 
          on=grouping_keys, 
          how='leftouter'
        )
    )

# fill-in any missing values
commodity_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_keys])

# COMMAND ----------

# DBTITLE 1,Write Features to Feature Store
# instantiate feature store client
fs = FeatureStoreClient()

# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f'{features_path}.commodity_features')
except: # create it now
  pass
  _ = (
    fs
      .create_table(
        name=f'{features_path}.commodity_features', # name of feature store table
        primary_keys= grouping_keys + ['select_date'], # name of keys that will be used to locate records
        schema=commodity_features.schema, # schema of feature set as derived from our feature_set dataframe
        description='commodity features used for propensity scoring' 
      )
    )

# merge feature set data into feature store
_ = (
  fs
    .write_table(
      name=f'{features_path}.commodity_features',
      df = commodity_features,
      mode = 'merge' # merge data into existing feature set, instead of 'overwrite'
    )
  )

# COMMAND ----------

# MAGIC %md ##Step 5: Generate Household-Commodity Features
# MAGIC
# MAGIC And now we can tackle the household-commodity combinations as follows:

# COMMAND ----------

# DBTITLE 1,Calculate Household-Commodity Features
# features will be grouped on households & commodities
grouping_keys = ['household_key','commodity_desc']

# get master set of household & commoditiy keys in incoming data
features = (
  household_commodity
    .select(grouping_keys)
    .distinct()
    .withColumn('select_date', fn.lit(select_date)) # assign date to feature set
    )

# calculate household-commodity features for each period and join to master set
for window in ['30d','60d','90d','1yr']:
  features = (
    features
      .join(
          get_features(transactions, grouping_keys, window=window), 
          on=grouping_keys, 
          how='leftouter'
        )
    )

# fill-in any missing values
household_commodity_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_keys])

# COMMAND ----------

# DBTITLE 1,Write Features to Feature Store
# instantiate feature store client
fs = FeatureStoreClient()

# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f'{features_path}.household_commodity_features')
except: # create it now
  fs.create_table(
    name=f'{features_path}.household_commodity_features', # name of feature store table
    primary_keys= grouping_keys + ['select_date'], # name of keys that will be used to locate records
    schema=household_commodity_features.schema, # schema of feature set as derived from our feature_set dataframe
    description='household-commodity features used for propensity scoring' 
  )

# merge feature set data into feature store

fs.write_table(
  name=f'{features_path}.household_commodity_features',
  df = household_commodity_features,
  mode = 'merge' # merge data into existing feature set, instead of 'overwrite'
)
