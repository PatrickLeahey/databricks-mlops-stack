import pytest
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, DoubleType, IntegerType

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
        ("2023-01-01", "household_1", "Commodity_A", "basket_1", "product_1", 10, 1, 0.5, 0, 0.5, 9.5),
        ("2023-01-02", "household_2", "Commodity_B", "basket_2", "product_2", 15, 2, 0, 1, 0, 14),
        # Add more rows as necessary to cover test scenarios
    ]
    schema = ["day", "household_key", "commodity_desc", "basket_id", "product_id", "amount_list", "instore_discount",
              "campaign_coupon_discount", "manuf_coupon_discount", "total_coupon_discount", "amount_paid"]
    return spark.createDataFrame(data, schema=schema)

@pytest.mark.parametrize("grouping_keys, window, expected_days", [
    (["household_key"], "30d", 30),
    (["commodity_desc"], "60d", 60),
    (["household_key", "commodity_desc"], "90d", 90)
])
def test_get_features_window_period(spark, transactional_data, grouping_keys, window, expected_days):
    result_df = get_features(transactional_data, grouping_keys, window)
    
    # Check if DataFrame is not empty
    assert result_df.count() > 0

    # Validate the days calculation based on window
    days_column = "days" + ("_" + window if window else "")
    result_days = result_df.select(days_column).collect()[0][0]
    assert result_days == expected_days, f"Expected {expected_days} days in window, got {result_days}"