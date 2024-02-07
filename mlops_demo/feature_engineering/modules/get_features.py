from datetime import timedelta
import pyspark.sql.functions as fn
from pyspark.sql.types import DoubleType

def get_features(df, grouping_keys, window=None):
  
  '''
  This function derives a number of features from our transactional data.
  
  df: the dataframe containing household transaction history

  grouping_keys: the household_key, commodity_desc or combination househould_key & commodity_desc around which to group data
  
  window: one of four supported string values:
    '30d': derive metrics from the last 30 days of the dataset
    '60d': derive metrics from the last 60 days of the dataset
    '90d': derive metrics from the last 90 days of the dataset
    '1yr': derive metrics from the 30 day period starting 1-year
           prior to the end of the dataset. this alings with the
           period from which our labels are derived.
  '''

  # get list of distinct grouping items in the original dataframe
  anchor_df = df.select(grouping_keys).distinct()
  
  # identify when dataset starts and ends
  min_day, max_day = (
    df
      .groupBy()
        .agg(
          fn.min('day').alias('min_day'), 
          fn.max('day').alias('max_day')
          )
      .collect()
    )[0]    
  
  ## print info to support validation
  #print('{0}:\t{1} days in original set between {2} and {3}'.format(window, (max_day - min_day).days + 1, min_day, max_day))
  
  # adjust min and max days based on specified window   
  if window == '30d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=30-1)
    
  elif window == '60d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=60-1)
    
  elif window == '90d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=90-1)
    
  elif window == '1yr':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=365-1)
    max_day = min_day + timedelta(days=30-1)
    
  else:
    raise Exception('unknown window definition')
  
  # determine the number of days in the window
  days_in_window = (max_day - min_day).days + 1
  
  ## print to help with date math validation
  #print('{0}:\t{1} days in adjusted set between {2} and {3}'.format(window, days_in_window, min_day, max_day))
  
  # convert dates to strings to make remaining steps easier
  max_day = max_day.strftime('%Y-%m-%d')
  min_day = min_day.strftime('%Y-%m-%d')
  
  # derive summary features from set
  summary_df = (
    df
      .filter(fn.expr(f"day between '{min_day}' and '{max_day}'")) # constrain to window
      .groupBy(grouping_keys)
        .agg(
          
          # summary metrics
          fn.countDistinct('day').alias('days'), 
          fn.countDistinct('basket_id').alias('baskets'),
          fn.count('product_id').alias('products'), 
          fn.count('*').alias('line_items'),
          fn.sum('amount_list').alias('amount_list'),
          fn.sum('instore_discount').alias('instore_discount'),
          fn.sum('campaign_coupon_discount').alias('campaign_coupon_discount'),
          fn.sum('manuf_coupon_discount').alias('manuf_coupon_discount'),
          fn.sum('total_coupon_discount').alias('total_coupon_discount'),
          fn.sum('amount_paid').alias('amount_paid'),
          
          # unique days with activity
          fn.countDistinct(
            fn.expr('case when instore_discount >0 then day else null end')
            ).alias('days_with_instore_discount'),
          fn.countDistinct(
            fn.expr('case when campaign_coupon_discount >0 then day else null end')
            ).alias('days_with_campaign_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when manuf_coupon_discount >0 then day else null end')
            ).alias('days_with_manuf_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when total_coupon_discount >0 then day else null end')
            ).alias('days_with_total_coupon_discount'),
          
          # unique baskets with activity
          fn.countDistinct(
            fn.expr('case when instore_discount >0 then basket_id else null end')
            ).alias('baskets_with_instore_discount'),
          fn.countDistinct(
            fn.expr('case when campaign_coupon_discount >0 then basket_id else null end')
            ).alias('baskets_with_campaign_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when manuf_coupon_discount >0 then basket_id else null end')
            ).alias('baskets_with_manuf_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when total_coupon_discount >0 then basket_id else null end')
            ).alias('baskets_with_total_coupon_discount'),          
    
          # unique products with activity
          fn.countDistinct(
            fn.expr('case when instore_discount >0 then product_id else null end')
            ).alias('products_with_instore_discount'),
          fn.countDistinct(
            fn.expr('case when campaign_coupon_discount >0 then product_id else null end')
            ).alias('products_with_campaign_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when manuf_coupon_discount >0 then product_id else null end')
            ).alias('products_with_manuf_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when total_coupon_discount >0 then product_id else null end')
            ).alias('products_with_total_coupon_discount'),          
    
          # unique line items with activity
          fn.sum(
            fn.expr('case when instore_discount >0 then 1 else null end')
            ).alias('line_items_with_instore_discount'),
          fn.sum(
            fn.expr('case when campaign_coupon_discount >0 then 1 else null end')
            ).alias('line_items_with_campaign_coupon_discount'),
          fn.sum(
            fn.expr('case when manuf_coupon_discount >0 then 1 else null end')
            ).alias('line_items_with_manuf_coupon_discount'),
          fn.sum(
            fn.expr('case when total_coupon_discount >0 then 1 else null end')
            ).alias('line_items_with_total_coupon_discount')          
          )    
    
      # per-day ratios
      .withColumn(
        f'baskets_per_day', 
        fn.expr('baskets/days')
        )
      .withColumn(
        f'products_per_day{window_suffix}', 
        fn.expr('products/days')
        )
      .withColumn(
        f'line_items_per_day', 
        fn.expr('line_items/days')
        )
      .withColumn(
        f'amount_list_per_day', 
        fn.expr('amount_list/days')
        )
      .withColumn(
        f'instore_discount_per_day', 
        fn.expr('instore_discount/days')
        )
      .withColumn(
        f'campaign_coupon_discount_per_day', 
        fn.expr('campaign_coupon_discount/days')
        )
      .withColumn(
        f'manuf_coupon_discount_per_day', 
        fn.expr('manuf_coupon_discount/days')
        )
      .withColumn(
        f'total_coupon_discount_per_day', 
        fn.expr('total_coupon_discount/days')
        )
      .withColumn(
        f'amount_paid_per_day', 
        fn.expr('amount_paid/days')
        )
      .withColumn(
        f'days_with_instore_discount_per_days', 
        fn.expr('days_with_instore_discount/days')
        )
      .withColumn(
        f'days_with_campaign_coupon_discount_per_days', 
        fn.expr('days_with_campaign_coupon_discount/days')
        )
      .withColumn(
        f'days_with_manuf_coupon_discount_per_days',
        fn.expr('days_with_manuf_coupon_discount/days')
        )
      .withColumn(
        f'days_with_total_coupon_discount_per_days', 
        fn.expr('days_with_total_coupon_discount/days')
        )
    
      # per-day-in-set ratios
      .withColumn(
        f'days_to_days_in_set', 
        fn.expr(f'days/{days_in_window}')
        )
      .withColumn(
        f'baskets_per_days_in_set', 
        fn.expr(f'baskets/{days_in_window}')
        )
      .withColumn(
        f'products_to_days_in_set', 
        fn.expr(f'products/{days_in_window}')
        )
      .withColumn(
        f'line_items_per_days_in_set', 
        fn.expr(f'line_items/{days_in_window}')
        )
      .withColumn(
        f'amount_list_per_days_in_set', 
        fn.expr(f'amount_list/{days_in_window}')
        )
      .withColumn(
        f'instore_discount_per_days_in_set', 
        fn.expr(f'instore_discount/{days_in_window}')
        )
      .withColumn(
        f'campaign_coupon_discount_per_days_in_set', 
        fn.expr(f'campaign_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'manuf_coupon_discount_per_days_in_set', 
        fn.expr(f'manuf_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'total_coupon_discount_per_days_in_set', 
        fn.expr(f'total_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'amount_paid_per_days_in_set', 
        fn.expr(f'amount_paid/{days_in_window}')
        )
      .withColumn(
        f'days_with_instore_discount_per_days_in_set', 
        fn.expr(f'days_with_instore_discount/{days_in_window}')
        )
      .withColumn(
        f'days_with_campaign_coupon_discount_per_days_in_set', 
        fn.expr(f'days_with_campaign_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'days_with_manuf_coupon_discount_per_days_in_set', 
        fn.expr(f'days_with_manuf_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'days_with_total_coupon_discount_per_days_in_set', 
        fn.expr(f'days_with_total_coupon_discount/{days_in_window}')
        )

      # per-basket ratios
      .withColumn(
        'products_per_basket', 
        fn.expr('products/baskets')
        )
      .withColumn(
        'line_items_per_basket', 
        fn.expr('line_items/baskets')
        )
      .withColumn(
        'amount_list_per_basket', 
        fn.expr('amount_list/baskets')
        )      
      .withColumn(
        'instore_discount_per_basket', 
        fn.expr('instore_discount/baskets')
        )  
      .withColumn(
        'campaign_coupon_discount_per_basket', 
        fn.expr('campaign_coupon_discount/baskets')
        ) 
      .withColumn(
        'manuf_coupon_discount_per_basket', 
        fn.expr('manuf_coupon_discount/baskets')
        )
      .withColumn(
        'total_coupon_discount_per_basket', 
        fn.expr('total_coupon_discount/baskets')
        )    
      .withColumn(
        'amount_paid_per_basket', 
        fn.expr('amount_paid/baskets')
        )
      .withColumn(
        'baskets_with_instore_discount_per_baskets', 
        fn.expr('baskets_with_instore_discount/baskets')
        )
      .withColumn(
        'baskets_with_campaign_coupon_discount_per_baskets', 
        fn.expr('baskets_with_campaign_coupon_discount/baskets')
        )
      .withColumn(
        'baskets_with_manuf_coupon_discount_per_baskets', 
        fn.expr('baskets_with_manuf_coupon_discount/baskets')
        )
      .withColumn(
        'baskets_with_total_coupon_discount_per_baskets', 
        fn.expr('baskets_with_total_coupon_discount/baskets')
        )
      
      # per-product ratios
      .withColumn(
        'line_items_per_product', 
        fn.expr('line_items/products')
        )
      .withColumn(
        'amount_list_per_product', 
        fn.expr('amount_list/products')
        )      
      .withColumn(
        'instore_discount_per_product', 
        fn.expr('instore_discount/products')
        )  
      .withColumn(
        'campaign_coupon_discount_per_product', 
        fn.expr('campaign_coupon_discount/products')
        ) 
      .withColumn(
        'manuf_coupon_discount_per_product', 
        fn.expr('manuf_coupon_discount/products')
        )
      .withColumn(
        'total_coupon_discount_per_product', 
        fn.expr('total_coupon_discount/products')
        )    
      .withColumn(
        'amount_paid_per_product', 
        fn.expr('amount_paid/products')
        )
      .withColumn(
        'products_with_instore_discount_per_product', 
        fn.expr('products_with_instore_discount/products')
        )
      .withColumn(
        'products_with_campaign_coupon_discount_per_product', 
        fn.expr('products_with_campaign_coupon_discount/products')
        )
      .withColumn(
        'products_with_manuf_coupon_discount_per_product', 
        fn.expr('products_with_manuf_coupon_discount/products')
        )
      .withColumn(
        'products_with_total_coupon_discount_per_product', 
        fn.expr('products_with_total_coupon_discount/products')
        )
      
      # per-line_item ratios
      .withColumn(
        'amount_list_per_line_item', 
        fn.expr('amount_list/line_items')
        )      
      .withColumn(
        'instore_discount_per_line_item', 
        fn.expr('instore_discount/line_items')
        )  
      .withColumn(
        'campaign_coupon_discount_per_line_item', 
        fn.expr('campaign_coupon_discount/line_items')
        ) 
      .withColumn(
        'manuf_coupon_discount_per_line_item', 
        fn.expr('manuf_coupon_discount/line_items')
        )
      .withColumn(
        'total_coupon_discount_per_line_item', 
        fn.expr('total_coupon_discount/line_items')
        )    
      .withColumn(
        'amount_paid_per_line_item', 
        fn.expr('amount_paid/line_items')
        )
      .withColumn(
        'products_with_instore_discount_per_line_item', 
        fn.expr('products_with_instore_discount/line_items')
        )
      .withColumn(
        'products_with_campaign_coupon_discount_per_line_item', 
        fn.expr('products_with_campaign_coupon_discount/line_items')
        )
      .withColumn(
        'products_with_manuf_coupon_discount_per_line_item', 
        fn.expr('products_with_manuf_coupon_discount/line_items')
        )
      .withColumn(
        'products_with_total_coupon_discount_per_line_item', 
        fn.expr('products_with_total_coupon_discount/line_items')
        )    
    
      # amount_list ratios
      .withColumn(
        'campaign_coupon_discount_to_amount_list', 
        fn.expr('campaign_coupon_discount/amount_list')
        )
      .withColumn(
        'manuf_coupon_discount_to_amount_list', 
        fn.expr('manuf_coupon_discount/amount_list')
        )
      .withColumn(
        'total_coupon_discount_to_amount_list', 
        fn.expr('total_coupon_discount/amount_list')
        )
      .withColumn(
        'amount_paid_to_amount_list', 
        fn.expr('amount_paid/amount_list')
        )
      )
 
  # derive days-since metrics
  dayssince_df = (
    df
      .filter(fn.expr(f"day <= '{max_day}'"))
      .groupBy(grouping_keys)
        .agg(
          fn.min(
            fn.expr(f"'{max_day}' - case when instore_discount >0 then day else '{min_day}' end")
            ).alias('days_since_instore_discount'),
          fn.min(
            fn.expr(f"'{max_day}' - case when campaign_coupon_discount >0 then day else '{min_day}' end")
            ).alias('days_since_campaign_coupon_discount'),
          fn.min(
            fn.expr(f"'{max_day}' - case when manuf_coupon_discount >0 then day else '{min_day}' end")
            ).alias('days_since_manuf_coupon_discount'),
          fn.min(
            fn.expr(f"'{max_day}' - case when total_coupon_discount >0 then day else '{min_day}' end"))
            .alias('days_since_total_coupon_discount')
          )
      )
  
  # combine metrics with anchor set to form return set 
  ret_df = (
    anchor_df
      .join(summary_df, on=grouping_keys, how='leftouter')
      .join(dayssince_df, on=grouping_keys, how='leftouter')
    )
  
  # rename fields based on control parameters
  for c in ret_df.columns:
    if c not in grouping_keys: # don't rename grouping fields
      ret_df = ret_df.withColumn(c, fn.col(c).cast(DoubleType())) # cast all metrics as doubles to avoid confusion as categoricals
      ret_df = ret_df.withColumnRenamed(c,f'{c}{window_suffix}')

  return ret_df