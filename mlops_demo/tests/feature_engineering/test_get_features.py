import pytest
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DateType, DoubleType, StringType

# Assuming your get_features function is located in a module named feature_engineering.py
from mlops_demo.feature_engineering.modules.get_features import get_features

@pytest.fixture(scope="session")
def spark(request):
    """Fixture for creating a Spark session."""
    spark = SparkSession.builder.master("local[1]").appName("pytest-pyspark-local-testing").getOrCreate()
    request.addfinalizer(lambda: spark.stop())
    return spark

@pytest.fixture
def transactional_data(spark):
    """Creates a DataFrame representing transactional data for testing."""
    data = [
        (1, 1, 1, 1, datetime.strptime("2023-01-01", "%Y-%m-%d"), 1200, 1, 10.0, 0.5, 0.0, 0.5, 0.5, 0.5, 9.5, 1, 1, "DEPARTMENT_A", "BRAND_A", "Commodity_A", "Sub_Commodity_A", "Size_A"),
        (2, 2, 2, 2, datetime.strptime("2023-01-02", "%Y-%m-%d"), 1300, 2, 15.0, 0.0, 1.0, 0.0, 1.0, 1.0, 14.0, 2, 2, "DEPARTMENT_B", "BRAND_B", "Commodity_B", "Sub_Commodity_B", "Size_B"),
    ]
    schema = StructType([
        StructField("product_id", IntegerType(), True),
        StructField("household_key", IntegerType(), True),
        StructField("basket_id", LongType(), True),
        StructField("week_no", IntegerType(), True),
        StructField("day", DateType(), True),
        StructField("trans_time", IntegerType(), True),
        StructField("store_id", IntegerType(), True),
        StructField("amount_list", DoubleType(), True),
        StructField("campaign_coupon_discount", DoubleType(), True),
        StructField("manuf_coupon_discount", DoubleType(), True),
        StructField("manuf_coupon_match_discount", DoubleType(), True),
        StructField("total_coupon_discount", DoubleType(), True),
        StructField("instore_discount", DoubleType(), True),
        StructField("amount_paid", DoubleType(), True),
        StructField("units", IntegerType(), True),
        StructField("MANUFACTURER", IntegerType(), True),
        StructField("DEPARTMENT", StringType(), True),
        StructField("BRAND", StringType(), True),
        StructField("COMMODITY_DESC", StringType(), True),
        StructField("SUB_COMMODITY_DESC", StringType(), True),
        StructField("CURR_SIZE_OF_PRODUCT", StringType(), True),
    ])
    return spark.createDataFrame(data, schema)

@pytest.mark.parametrize("grouping_keys, window", [
    (["household_key"], "30d"),
    (["commodity_desc"], "60d"),
    (["household_key", "commodity_desc"], "90d")
])
def test_get_features_window_period(spark, transactional_data, grouping_keys, window):
    result_df = get_features(transactional_data, grouping_keys, window)
    
    # Check if DataFrame is not empty
    assert result_df.count() > 0