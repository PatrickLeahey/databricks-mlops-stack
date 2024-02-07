# Databricks notebook source
# MAGIC %md The purpose of this notebook is to train the models required for our propensity scoring work. This notebook was developed using the **Databricks 13.3 LTS ML** runtime.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will provide the logic needed to retrain the models for each of our product commodity (categories).  For each commodity, we will tune the model before training a final instance that will be immediately elevated to be the production instance of the propensity model for that category.
# MAGIC
# MAGIC **NOTE** Before running this notebook, make sure you have populated the feature store with features from 30 days back from the *current day*.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

import databricks.feature_store as feature_store
from databricks.feature_store import FeatureStoreClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

import pyspark.sql.functions as fn

from datetime import datetime, timedelta, date
import pathlib

# COMMAND ----------

# MAGIC %md ##Step 1: Retrieve Configuration Settings
# MAGIC
# MAGIC This notebook will be typically run as part of a regularly scheduled workflow.  However, during development and initialization of the feature store, we should expect to see it run manually.  To support either scenario, we will define [widgets](https://docs.databricks.com/notebooks/widgets.html) through which values can be assigned to the notebook, either directly or through a runtime call from another notebook.  We will first attempt to retrieve configuration values from the jobs workflow but failing that, we will fallback to values supplied by these widgets:

# COMMAND ----------

# DBTITLE 1,Create Widgets
dbutils.widgets.text('select_date', defaultValue=date.today().strftime('%Y-%m-%d'), label="Select Date")
dbutils.widgets.text('input_data_catalog', defaultValue='pl_mlops_demo_prod', label="Input Data Catalog")
dbutils.widgets.text('input_data_schema', defaultValue='mlops_demo', label="Input Data Schema")
dbutils.widgets.text('features_catalog', defaultValue='pl_mlops_demo_dev', label="Features Catalog")
dbutils.widgets.text('features_schema', defaultValue='mlops_demo', label="Features Schema")
dbutils.widgets.text('experiment_name', defaultValue='/Users/patrick.leahey@databricks.com/propensity', label="Experiment Name")
dbutils.widgets.text('model_name', defaultValue='pl_mlops_demo_dev.mlops_demo.propensity', label="Model Name")
dbutils.widgets.text('commodity', defaultValue='EGGS', label="Commodity")

# COMMAND ----------

# DBTITLE 1,Get Widget Values
select_date = '2019-10-31' # select_date = dbutils.widgets.get('select_date')
input_data_catalog = dbutils.widgets.get('input_data_catalog')
input_data_schema = dbutils.widgets.get('input_data_schema')
features_catalog = dbutils.widgets.get('features_catalog')
features_schema = dbutils.widgets.get('features_schema')
experiment_name = dbutils.widgets.get('experiment_name')
model_name = dbutils.widgets.get('model_name')
commodity = dbutils.widgets.get('commodity')

select_date = datetime.strptime(select_date, '%Y-%m-%d').date()
input_data_path = f'{input_data_catalog}.{input_data_schema}'
features_path = f'{features_catalog}.{features_schema}'

assert experiment_name
assert model_name
assert commodity

# COMMAND ----------

# MAGIC %md ##Step 2: Determine Date Ranges
# MAGIC
# MAGIC With the *select date* known, we now can retrieve features and derive labels. The select date is important as this represents the latest point from which we can train a model.  In our propensity scoring scenario, we envision making a prediction for likelihood to purchase over the next 30 days.  To train a model for this, we must derive a label using data 30-days back and up to the select date.  Features used to then predict that label must be derived from data prior to this.  We might understand the relationship between the select date and days prior during model training as follows:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/aiq_days_prior3.png' width=60%>
# MAGIC
# MAGIC </p>
# MAGIC With this in mind, we might define the start and end of our label and feature inputs as follows:

# COMMAND ----------

# DBTITLE 1,Define Cutoff Days for Features and Labels
labels_end_day = select_date
features_end_day = labels_end_day - timedelta(days=(30))

labels_start_day = features_end_day + timedelta(days=1)

print(f"We will derive features from the start of the dataset through {features_end_day}.")
print(f"We will derive labels from {labels_start_day} through {labels_end_day}, i.e. {(labels_end_day - labels_start_day).days + 1} days")

# COMMAND ----------

# MAGIC %md ##Step 2: Assemble Labels
# MAGIC
# MAGIC With the date ranges for labels defined, we can now derive labels for each household and commodity by first retrieving commodities with purchases by a given household within our label creation period.  These will be our positive class labels:

# COMMAND ----------

# DBTITLE 1,Identify Household-Commodity Pairs Positive in Label Period
positive_labels = (
  spark
    .table(f'{input_data_path}.transactions_adj')
    .filter(fn.expr(f"day BETWEEN '{labels_start_day}' AND '{labels_end_day}'")) # in label period
    .join(spark.table(f'{input_data_path}.products'), on='product_id')
    .filter(fn.col('commodity_desc')==commodity)
    .select('household_key','commodity_desc') # households and commodities that saw a purchase in period
    .distinct()
    .withColumn('purchased', fn.lit(1)) # these are the positive labels
  )

# COMMAND ----------

# MAGIC %md We can then grab every household-commodity combination we could likely see in this same period:

# COMMAND ----------

# DBTITLE 1,Identify All Household-Commodity Combinations in Dataset
# get unique households
households = spark.table(f'{input_data_path}.transactions_adj').select('household_key').distinct()

# COMMAND ----------

# MAGIC %md Combining these with a left-outer join, we can flag those that received a purchase with a label of 1 and those that did not with a label of 0:

# COMMAND ----------

# DBTITLE 1,Combine with Positive Labels to Determine Negative Labels
labels = (
  households
    .join(
      positive_labels, 
      on=['household_key'], 
      how='leftouter'
      )
    .withColumn('select_date', fn.lit(features_end_day))
    .withColumn('purchased', fn.expr("coalesce(purchased, 0)"))
    .withColumn('commodity_desc', fn.lit('EGGS'))
    .orderBy('household_key')
  ).cache()

display(labels)

# COMMAND ----------

# MAGIC %md ##Step 3: Retrieve Features
# MAGIC
# MAGIC We can now retrieve our features as they existed the day prior to the start of our label calculation period. Because these features were previously calculated and retained in the feature store, we can retrieve them as follows:

# COMMAND ----------

# DBTITLE 1,Define Feature Retrieval Logic (Feature Lookups)
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

# MAGIC %md ##Step 4: Define Model Training Functions
# MAGIC
# MAGIC With our feature and label data retrieved, we can now launch the process to train our models.  For each model, we'll perform a hyperparameter tuning run followed by a final model training cycle.  The logic for performing a hyperparameter tuning run will be defined as follows, where the metric that is to serve as the focus of our model tuning exercise is returned as a loss value that we seek to minimize:

# COMMAND ----------

# DBTITLE 1,Define Function to Train Model Given a Set of Hyperparameter Values
def evaluate_model (hyperopt_params):
  
  # accesss replicated input data
  _X_train = X_train_broadcast.value
  _y_train = y_train_broadcast.value
  _X_validate = X_validate_broadcast.value
  _y_validate = y_validate_broadcast.value
  
  # configure model parameters
  params = hyperopt_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  # instantiate model with parameters
  model = XGBClassifier(
    enable_categorical=True,
    tree_method='hist',
    **params
  )
  
  # train
  model.fit(X_train, y_train)
  
  # predict
  y_pred = model.predict(X_validate)
  y_prob = model.predict_proba(X_validate)
  
  # eval metrics
  model_ap = average_precision_score(y_validate, y_prob[:,1])
  model_ba = balanced_accuracy_score(y_validate, y_pred)
  model_mc = matthews_corrcoef(y_validate, y_pred)
  
  # log metrics with mlflow run
  mlflow.log_metrics({
    'avg precision':model_ap,
    'balanced_accuracy':model_ba,
    'matthews corrcoef':model_mc
    })                                       
                                             
  # invert key metric for hyperopt
  loss = -1 * model_ap
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md Similarly, we can define a function to train our model based on the discovered set of *best* hyperparameter values.  Note that this function's logic closely mirrors that of the function used for hyperparameter tuning with the exception that our return values differ and we intend to train this final model on our cluster's driver and not on the worker nodes (so that we do not need to mess with broadcasted datasets):

# COMMAND ----------

# DBTITLE 1,Define Function to Train Model Given Best Hyperparameter Values
def train_final_model (hyperopt_params):
   
  # configure model parameters
  params = hyperopt_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  # instantiate model with parameters
  model = XGBClassifier(
    enable_categorical=True,
    tree_method='hist',
    **params
  )
  
  # train
  model.fit(X_train_validate, y_train_validate)

  return model

# COMMAND ----------

# MAGIC %md ##Step 5: Train Per-Commodity
# MAGIC
# MAGIC We can now tune and train models for commodity we wish to score

# COMMAND ----------

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')
client = MlflowClient(registry_uri="databricks-uc")

def get_latest_model_version(model_name):
  latest_version = 1
  for mv in client.search_model_versions(f"name='{model_name}'"):
      version_int = int(mv.version)
      if version_int > latest_version:
          latest_version = version_int
  return latest_version

# COMMAND ----------

# DBTITLE 1,Tune & Train Model
# instantiate feature store client
fs = FeatureStoreClient()

training_set = fs.create_training_set(
  df=labels,
  feature_lookups=feature_lookups,
  label='purchased',
  exclude_columns=['household_key','select_date','commodity_desc']
)

# get features and labels
features_and_labels = training_set.load_df().toPandas()

X = features_and_labels.drop('purchased', axis=1)
y = features_and_labels['purchased']

# split into train (0.70), validate (0.15) and test (0.15)
X_train_validate, X_test,  y_train_validate, y_test = train_test_split(X, y, test_size=0.15)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, test_size=(0.15/0.85))

# broadcast sets
X_train_broadcast = sc.broadcast(X_train)
y_train_broadcast = sc.broadcast(y_train)
X_validate_broadcast = sc.broadcast(X_validate)
y_validate_broadcast = sc.broadcast(y_validate)

# COMMAND ----------

search_space = {
  'max_depth' : hp.quniform('max_depth', 5, 20, 1),
  'learning_rate' : hp.uniform('learning_rate', 0.01, 0.40)
}

# determine pos_class_weight
pos_class_weight = labels.count()/labels.filter('purchased=1').count()
if pos_class_weight > 1.0:
  search_space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 1.0, 5 * pos_class_weight)

# get at least 50 trials in but ideally 5 per node
max_evals = max(50, sc.defaultParallelism*5)

argmin = None
# perform tuning
with mlflow.start_run(run_name='tuning'):
  # try hyperparameter tuning
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=SparkTrials(parallelism=sc.defaultParallelism)
  )

# COMMAND ----------

argmin

# COMMAND ----------

with mlflow.start_run(run_name='training') as run:
  model = train_final_model(argmin)

  predictions = model.predict(X_test)
  signature = infer_signature(model_input=X_test, model_output=predictions)

  model_info = mlflow.sklearn.log_model(
    model,
    artifact_path='model',
    registered_model_name=model_name,
    signature=signature
  )

  model_version = get_latest_model_version(model_name)

# COMMAND ----------

eval_data = X_test
eval_data['target'] = y_test

challenger_eval = mlflow.evaluate(
  model_info.model_uri,
  eval_data,
  targets='target',
  model_type='classifier',
  evaluators=['default']
)

auc_benchmark = 0.75
if challenger_eval.metrics['roc_auc'] >= auc_benchmark:
  client.set_registered_model_alias(model_name, "Challenger", model_version)
  print(f'Model models:/{model_name}/{model_version} passed benchmarks, promoted to Challenger')

  try:
    champion_eval = mlflow.evaluate(
      f'models:/{model_name}@Champion',
      eval_data,
      targets='target',
      model_type='classifier',
      evaluators=['default']
    )

    if challenger_eval.metrics['roc_auc'] >= champion_eval.metrics['roc_auc']:
      client.delete_registered_model_alias(model_name, 'Challenger')
      client.set_registered_model_alias(model_name, 'Champion', model_version)
      print(f'Challenger models:/{model_name}/{model_version} beat existing Champion, promoted to Champion')
      

  except:
    client.delete_registered_model_alias(model_name, 'Challenger')
    client.set_registered_model_alias(model_name, 'Champion', model_version)
    print(f'Champion model version does not exist, promoted models:/{model_name}/{model_version}')

else:
  print(f'Model models:/{model_name}/{model_version} did not pass benchmarks -> AUC: {challenger_eval.metrics["roc_auc"]} < {auc_benchmark}')

# COMMAND ----------

challenger_eval.metrics
