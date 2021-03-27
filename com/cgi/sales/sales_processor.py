from os import path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F


class SalesProcessor:
    """
    Class in charge of process the sales

    Attributes
    ----------
    spark : SparkSession
        a Spark session
    dataset_path : str
        sales dataset input path. It can be in any format supported by Spark, i.e. local absolute path, S3 path.
    output_path : str
        processing results output path. It can be in any format supported by Spark, i.e. local absolute path, S3 path.
    """

    def __init__(self, spark: SparkSession, dataset_path: str, output_path: str):
        """
        Parameters
        ----------
        :param spark: SparkSession
            a Spark session
        :param dataset_path: str
            sales dataset input path. It can be any format supported by Spark, i.e. local absolute path, S3 path.
        :param output_path: str
             processing results output path. It can be any format supported by Spark, i.e. local absolute path, S3 path.
        """
        self.spark = spark
        self.dataset_path = dataset_path
        self.output_path = output_path

    def run(self):
        """
        This method runs the sales processor reading from the dataset path and writing the processing results to the
        output path
        """
        print(f'Dataset path: {self.dataset_path}')
        sales_path = f'{self.dataset_path}/sales_train_validation.csv'
        print(f'Sales path: {sales_path}')
        sell_prices_path = f'{self.dataset_path}/sell_prices.csv'
        print(f'Sell prices path: {sell_prices_path}')
        calendar_path = f'{self.dataset_path}/calendar.csv'
        print(f'Calendar path: {calendar_path}')
        print(f'Output path: {self.output_path}')

        # Reading the sales dataset
        sales: DataFrame = self.read_csv_with_headers(self.spark, sales_path)
        # Number or day to process
        n_days_to_keep = 100
        # Total day available in the dataset
        total_days = 1913

        columns_to_drop = self.get_day_columns_to_drop(n_days_to_keep, total_days)
        columns_to_drop.extend(['id', 'dept_id', 'cat_id', 'state_id'])
        sales_n_days: DataFrame = self.drop_columns(columns_to_drop, sales)
        """
        sales_n_days:
        
        Count: 30.490  
              
        +-------------+--------+---+---+---+---+---+---+---+---+---+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+-----+
        |item_id      |store_id|d_1|d_2|d_3|d_4|d_5|d_6|d_7|d_8|d_9|d_10|d_11|d_12|d_13|d_14|d_15|d_16|d_17|d_18|d_19|d_20|d_21|d_22|d_23|d_24|d_25|d_26|d_27|d_28|d_29|d_30|d_31|d_32|d_33|d_34|d_35|d_36|d_37|d_38|d_39|d_40|d_41|d_42|d_43|d_44|d_45|d_46|d_47|d_48|d_49|d_50|d_51|d_52|d_53|d_54|d_55|d_56|d_57|d_58|d_59|d_60|d_61|d_62|d_63|d_64|d_65|d_66|d_67|d_68|d_69|d_70|d_71|d_72|d_73|d_74|d_75|d_76|d_77|d_78|d_79|d_80|d_81|d_82|d_83|d_84|d_85|d_86|d_87|d_88|d_89|d_90|d_91|d_92|d_93|d_94|d_95|d_96|d_97|d_98|d_99|d_100|
        +-------------+--------+---+---+---+---+---+---+---+---+---+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+-----+
        |HOBBIES_1_001|CA_1    |0  |0  |0  |0  |0  |0  |0  |0  |0  |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0   |0    |
        +-------------+--------+---+---+---+---+---+---+---+---+---+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+-----+
        """
        n_day_columns = self.get_n_days_columns(n_days_to_keep)
        """
        Problem 1:
        ----------        
        Considering only the first 100 days of sales, which were the 5 stores that produced more revenue. 
        """
        sales_n_days_exploded = self.get_sales_n_days_exploded(n_day_columns, sales_n_days)
        """
        sales_n_days_exploded:
        
        Count: 3.049.000
        """
        sales_n_days_non_zero = self.non_zero_sales(sales_n_days_exploded).cache()
        """
        sales_n_days_non_zero:
        
        Count: 645.488
        
        | -- item_id: string(nullable=true)
        | -- store_id: string(nullable=true)
        | -- day: string(nullable=false)
        | -- unit_sales: integer(nullable=true)
        
        +-------------+--------+----+----------+
        |item_id      |store_id|day |unit_sales|
        +-------------+--------+----+----------+
        |HOBBIES_1_004|CA_1    |d_37|2         |
        +-------------+--------+----+----------+
        """
        calendar: DataFrame = self.read_csv_with_headers(self.spark, calendar_path)

        calendar_day_and_week = self.get_calendar_day_and_week(calendar)
        """
        calendar_day_and_week:
        
        Count: 1.969
        
        root
        | -- week: integer(nullable=true)
        | -- day: string(nullable=true)
        
        +-----+---+
        |week |day|
        +-----+---+
        |11101|d_1|
        +-----+---+
        """
        max_week = self.get_max_week_until_n_days(calendar_day_and_week, n_days_to_keep)
        calendar_until_week = self.filter_until_week(calendar_day_and_week, max_week)
        sales_n_days_with_week = self.get_sales_with_week(calendar_until_week, sales_n_days_non_zero)
        """
        sales_n_days_with_week:
        
        +----+-------------+--------+----------+-----+
        |day |item_id      |store_id|unit_sales|week |
        +----+-------------+--------+----------+-----+
        |d_37|HOBBIES_1_004|CA_1    |2         |11106|
        +----+-------------+--------+----------+-----+
        """
        prices: DataFrame = self.read_csv_with_headers(self.spark, sell_prices_path) \
            .withColumnRenamed('wm_yr_wk', 'week')
        """
        prices:
        
        Count: 6.841.121

        +--------+-------------+-----+----------+
        |store_id|item_id      |week |sell_price|
        +--------+-------------+-----+----------+
        |CA_1    |HOBBIES_1_001|11325|9.58      |
        +--------+-------------+--------+----------+
        """
        prices_until_week = self.filter_until_week(prices, max_week)
        """
        prices_until_week:
        
        Count: 203.088
        """
        sales_n_days_with_prices = self.get_sales_with_prices(prices_until_week, sales_n_days_with_week)
        """
        sales_n_days_with_prices:
        
        Count: 645.488
        
        root
         |-- store_id: string (nullable = true)
         |-- item_id: string (nullable = true)
         |-- week: integer (nullable = true)
         |-- day: string (nullable = false)
         |-- unit_sales: integer (nullable = true)
         |-- sell_price: double (nullable = true)
        
        +--------+-----------+-----+---+----------+----------+
        |store_id|item_id    |week |day|unit_sales|sell_price|
        +--------+-----------+-----+---+----------+----------+
        |CA_1    |FOODS_1_069|11101|d_1|2         |2.18      |
        +--------+-----------+-----+---+----------+----------+
        """
        sales_n_days_with_total_prices = self.get_sales_with_total_prices(sales_n_days_with_prices)
        """
        sales_n_days_with_total_prices:
        
        +--------+-----------+--------+---+----------+----------+-----------+
        |store_id|item_id    |week    |day|unit_sales|sell_price|total_price|
        +--------+-----------+--------+---+----------+----------+-----------+
        |CA_1    |FOODS_1_069|11101   |d_1|2         |2.18      |4.36       |
        +--------+-----------+--------+---+----------+----------+-----------+
        """
        # Number of stores to rank
        n_stores_to_rank = 5
        top_n_stores_by_revenue = self.get_top_n_stores_by_revenue(n_stores_to_rank, sales_n_days_with_total_prices)
        """
        Solution to problem 1:
        ----------------------
        
        +--------+------------------+----------+
        |store_id|store_revenue     |store_rank|
        +--------+------------------+----------+
        |CA_3    |1129280.9600000007|1         |
        |CA_1    |838689.3799999995 |2         |
        |TX_2    |810773.2599999995 |3         |
        |WI_3    |809274.1300000002 |4         |
        |CA_2    |698924.3999999992 |5         |
        +--------+------------------+----------+
        """
        # Writing the result to csv
        self.overwrite_csv_with_headers(top_n_stores_by_revenue, path.join(self.output_path, 'top_n_stores_by_revenue'))
        """
        Problem 2:
        ----------
        Considering only the first 100 days of sales, for each store what were the two best selling products.
        """
        sales_n_days_by_store_and_item = self.get_sales_n_days_by_store_and_item(n_day_columns, sales_n_days)
        """
        sales_n_days_by_store_and_item:
        
        Count: 30.490
        
        +--------+-------------+----------------+
        |store_id|item_id      |sum_n_days_sales|
        +--------+-------------+----------------+
        |CA_1    |HOBBIES_1_001|0               |
        +--------+-------------+----------------+
        """
        # Number of items to rank per store.
        n_items_to_rank = 2
        top_n_items_sold_by_store = self.get_top_n_items_sold_by_store(n_items_to_rank, sales_n_days_by_store_and_item)
        """
        Solution to problem 2:
        ----------------------
        
        +--------+-----------+---------------------+---------+
        |store_id|item_id    |sum_n_days_item_sales|item_rank|
        +--------+-----------+---------------------+---------+
        |WI_2    |FOODS_3_376|4025                 |1        |
        |WI_2    |FOODS_3_694|3316                 |2        |
        |WI_3    |FOODS_3_586|5560                 |1        |
        |WI_3    |FOODS_3_694|4600                 |2        |
        |TX_2    |FOODS_3_586|8536                 |1        |
        |TX_2    |FOODS_3_555|5022                 |2        |
        |WI_1    |FOODS_3_226|3511                 |1        |
        |WI_1    |FOODS_3_694|2343                 |2        |
        |TX_1    |FOODS_3_586|6089                 |1        |
        |TX_1    |FOODS_3_555|4143                 |2        |
        |CA_4    |FOODS_3_587|2176                 |1        |
        |CA_4    |FOODS_3_635|1330                 |2        |
        |TX_3    |FOODS_3_586|9966                 |1        |
        |TX_3    |FOODS_3_555|6421                 |2        |
        |CA_2    |FOODS_3_586|3274                 |1        |
        |CA_2    |FOODS_3_252|2556                 |2        |
        |CA_1    |FOODS_3_587|6890                 |1        |
        |CA_1    |FOODS_3_714|4079                 |2        |
        |CA_3    |FOODS_3_587|8335                 |1        |
        |CA_3    |FOODS_3_586|6216                 |2        |
        +--------+-----------+---------------------+---------+
        """
        # Writing the result to csv
        self.overwrite_csv_with_headers(top_n_items_sold_by_store,
                                        path.join(self.output_path, 'top_n_items_sold_by_store'))

    @staticmethod
    def get_n_days_columns(n_days_to_keep):
        return [f'd_{i}' for i in range(1, n_days_to_keep + 1)]

    @staticmethod
    def read_csv_with_headers(spark, path):
        return spark.read.option('inferSchema', True).option('header', True).csv(path)

    @staticmethod
    def get_day_columns_to_drop(n_days_to_keep, max_day):
        return [f'd_{i}' for i in range(n_days_to_keep + 1, max_day + 1)]

    @staticmethod
    def drop_columns(columns_to_drop, df):
        return df.drop(*columns_to_drop)

    @staticmethod
    def get_sales_n_days_exploded(n_day_columns, sales_n_days):
        day_sales_col = F.array(
            *[F.struct(F.lit(day_col_name).alias('day'), F.col(day_col_name).alias('unit_sales'))
              for day_col_name in n_day_columns])
        sales_n_days_exploded = sales_n_days.withColumn('day_sales', F.explode(day_sales_col)) \
            .drop(*n_day_columns) \
            .withColumn('day', F.col('day_sales.day')) \
            .withColumn('unit_sales', F.col('day_sales.unit_sales')) \
            .drop('day_sales')
        return sales_n_days_exploded

    @staticmethod
    def non_zero_sales(sales_n_days_exploded):
        return sales_n_days_exploded.filter('unit_sales > 0')

    @staticmethod
    def get_calendar_day_and_week(calendar):
        return calendar \
            .select(['wm_yr_wk', 'd']) \
            .withColumnRenamed('wm_yr_wk', 'week') \
            .withColumnRenamed('d', 'day')

    @staticmethod
    def get_max_week_until_n_days(calendar_day_by_week, n_days_to_keep):
        return calendar_day_by_week \
            .filter(f"day = 'd_{n_days_to_keep}'").select('week').first()['week']

    @staticmethod
    def filter_until_week(df, max_week):
        return df.filter(f'week <= {max_week}')

    @staticmethod
    def get_sales_with_week(calendar, sales):
        return sales.join(calendar, 'day', 'left')

    @staticmethod
    def get_sales_with_prices(prices, sales):
        return sales.join(prices,
                          ['store_id', 'item_id', 'week'], 'left')

    @staticmethod
    def get_sales_with_total_prices(sales_n_days_with_prices):
        return sales_n_days_with_prices \
            .withColumn('total_price', F.col('unit_sales') * F.col('sell_price'))

    @staticmethod
    def get_top_n_stores_by_revenue(n_stores_to_rank, sales):
        revenue_by_stores_window = Window.orderBy(F.col('store_revenue').desc())
        top_n_stores_revenue = sales.groupBy('store_id') \
            .agg(F.sum('total_price').alias('store_revenue')) \
            .select('*', F.rank().over(revenue_by_stores_window).alias('store_rank')) \
            .filter(F.col('store_rank') <= n_stores_to_rank)
        return top_n_stores_revenue

    @staticmethod
    def get_sales_n_days_by_store_and_item(n_day_columns, sales):
        n_days_item_sales_list_col = F.array(*[F.col(day_col_name) for day_col_name in n_day_columns])
        acc_expr = 'cast(0 as int)'
        sum_function_expr = '(acc, x) -> acc + x'
        return sales.withColumn('n_days_item_sales_list', n_days_item_sales_list_col) \
            .drop(*n_day_columns) \
            .withColumn('sum_n_days_item_sales',
                        F.expr(f'AGGREGATE(n_days_item_sales_list, {acc_expr}, {sum_function_expr})')) \
            .select(['store_id', 'item_id', 'sum_n_days_item_sales'])

    @staticmethod
    def get_top_n_items_sold_by_store(n_items_to_rank, sales_n_days_by_store_and_item):
        item_sales_by_store_window = Window.partitionBy(F.col('store_id')).orderBy(
            F.col('sum_n_days_item_sales').desc())
        top_n_items_sold_by_store = sales_n_days_by_store_and_item \
            .select('*', F.rank().over(item_sales_by_store_window).alias('item_rank')) \
            .filter(F.col('item_rank') <= n_items_to_rank)
        return top_n_items_sold_by_store

    @staticmethod
    def overwrite_csv_with_headers(dataframe: DataFrame, output_path: str):
        dataframe.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
