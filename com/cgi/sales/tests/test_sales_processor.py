import shutil
import tempfile
from os import path

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

from com.cgi.sales.sales_processor import SalesProcessor
from com.cgi.sales.tests.test_pyspark import PySparkTest


class TestSalesProcessor(PySparkTest):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_read_csv_with_headers(self):
        # given
        data_path = path.join(self.test_dir, 'my_data.csv')
        f = open(data_path, 'w')
        # headers
        f.write('foo,bar\n')
        # values
        f.write('0,a')
        f.close()
        schema = StructType([
            StructField("foo", IntegerType(), True),
            StructField("bar", StringType(), True),
        ])
        expected = self.spark.createDataFrame(data=[(0, 'a')], schema=schema)
        # when
        actual = SalesProcessor.read_csv_with_headers(self.spark, data_path)
        # then
        self.assert_dataframe_equals(actual, expected)

    @staticmethod
    def test_get_day_columns_to_drop():
        # given
        expected = ['d_3', 'd_4', 'd_5']
        # when
        columns_to_drop = SalesProcessor.get_day_columns_to_drop(2, 5)
        # then
        assert expected == columns_to_drop

    def test_drop_columns(self):
        # given
        schema = StructType([
            StructField("foo", IntegerType(), True),
            StructField("bar", StringType(), True),
        ])
        df = self.spark.createDataFrame(data=[(0, 'a')], schema=schema)
        expected_schema = StructType([
            StructField("foo", IntegerType(), True),
        ])
        expected = self.spark.createDataFrame(data=[(0,)], schema=expected_schema)
        # when
        actual = SalesProcessor.drop_columns(['bar'], df)
        # then
        self.assert_dataframe_equals(actual, expected)

    @staticmethod
    def test_get_n_days_columns():
        # given
        n_days = 5
        expected = ['d_1', 'd_2', 'd_3', 'd_4', 'd_5']
        # when
        n_day_columns = SalesProcessor.get_n_days_columns(n_days)
        # then
        assert expected == n_day_columns

    def test_get_sales_n_days_exploded(self):
        # given
        schema = StructType([
            StructField("item_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("d_1", IntegerType(), True),
            StructField("d_2", IntegerType(), True),
        ])
        data = [
            ('HOBBIES_1_001', 'CA_1', 1, 2),
            ('HOBBIES_2_002', 'CA_2', 3, 4)
        ]
        df = self.spark.createDataFrame(data=data, schema=schema)

        expected_schema = StructType([
            StructField("item_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
        ])
        expected_data = [
            ('HOBBIES_1_001', 'CA_1', 'd_1', 1),
            ('HOBBIES_1_001', 'CA_1', 'd_2', 2),
            ('HOBBIES_2_002', 'CA_2', 'd_1', 3),
            ('HOBBIES_2_002', 'CA_2', 'd_2', 4)
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        n_days_columns = ['d_1', 'd_2']
        # when
        actual = SalesProcessor.get_sales_n_days_exploded(n_days_columns, df)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_non_zero_sales(self):
        # given
        schema = StructType([
            StructField("item_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
        ])
        data = [
            ('HOBBIES_1_001', 'CA_1', 'd_1', 0),
            ('HOBBIES_1_001', 'CA_1', 'd_2', 2),
            ('HOBBIES_2_002', 'CA_2', 'd_1', 0),
            ('HOBBIES_2_002', 'CA_2', 'd_2', 4)
        ]
        df = self.spark.createDataFrame(data=data, schema=schema)

        expected_data = [
            ('HOBBIES_1_001', 'CA_1', 'd_2', 2),
            ('HOBBIES_2_002', 'CA_2', 'd_2', 4)
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=schema)
        # when
        actual = SalesProcessor.non_zero_sales(df)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_calendar_days_and_weeks(self):
        # given
        schema = StructType([
            StructField("wm_yr_wk", IntegerType(), True),
            StructField("d", StringType(), True),
            StructField("foo", StringType(), True),
        ])
        data = [
            (11101, 'd_1', 'bar'),
        ]
        df = self.spark.createDataFrame(data=data, schema=schema)
        expected_schema = StructType([
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
        ])
        expected_data = [
            (11101, 'd_1'),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        # when
        actual = SalesProcessor.get_calendar_day_and_week(df)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_max_week_until_n_days(self):
        # given
        schema = StructType([
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
        ])
        data = [
            (11101, 'd_1'),
            (11102, 'd_2'),
            (11103, 'd_3'),
        ]
        df = self.spark.createDataFrame(data=data, schema=schema)
        n_days_to_keep = 2
        expected = 11102
        # when
        actual = SalesProcessor.get_max_week_until_n_days(df, n_days_to_keep)
        # then
        assert expected == actual

    def test_get_calendar_until_week(self):
        # given
        max_week = 11101
        schema = StructType([
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
        ])
        data = [
            (max_week, 'd_7'),
            (max_week, 'd_8'),
            (max_week + 1, 'd_9'),
        ]
        df = self.spark.createDataFrame(data=data, schema=schema)
        expected_data = [
            (max_week, 'd_7'),
            (max_week, 'd_8'),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=schema)
        # when
        actual = SalesProcessor.filter_until_week(df, max_week)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_sales_with_week(self):
        # given
        calendar_schema = StructType([
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
        ])
        calendar_data = [
            (1, 'd_7'),
            (1, 'd_8'),
            (2, 'd_9'),
        ]
        calendar = self.spark.createDataFrame(data=calendar_data, schema=calendar_schema)

        sales_schema = StructType([
            StructField("item_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
        ])
        sales_data = [
            ('HOBBIES_1_001', 'CA_1', 'd_7', 3),
            ('HOBBIES_1_001', 'CA_1', 'd_8', 4),
            ('HOBBIES_2_002', 'CA_2', 'd_9', 5),
        ]
        sales = self.spark.createDataFrame(data=sales_data, schema=sales_schema)

        expected_schema = StructType([
            StructField("day", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
            StructField("week", IntegerType(), True),
        ])
        expected_data = [
            ('d_7', 'HOBBIES_1_001', 'CA_1', 3, 1),
            ('d_8', 'HOBBIES_1_001', 'CA_1', 4, 1),
            ('d_9', 'HOBBIES_2_002', 'CA_2', 5, 2),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        # when
        actual = SalesProcessor.get_sales_with_week(calendar, sales)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_sales_with_prices(self):
        # given
        prices_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("week", IntegerType(), True),
            StructField("sell_price", DoubleType(), True),
        ])
        prices_data = [
            ('STORE_1', 'ITEM_1', 1, 1.5),
            ('STORE_2', 'ITEM_2', 2, 3.5),
        ]
        prices = self.spark.createDataFrame(data=prices_data, schema=prices_schema)

        sales_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
        ])
        sales_data = [
            ('STORE_1', 'ITEM_1', 1, 'd_7', 3),
            ('STORE_2', 'ITEM_2', 2, 'd_8', 4),
        ]
        sales = self.spark.createDataFrame(data=sales_data, schema=sales_schema)

        expected_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
            StructField("sell_price", DoubleType(), True),
        ])
        expected_data = [
            ('STORE_1', 'ITEM_1', 1, 'd_7', 3, 1.5),
            ('STORE_2', 'ITEM_2', 2, 'd_8', 4, 3.5),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        # when
        actual = SalesProcessor.get_sales_with_prices(prices, sales)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_sales_with_total_prices(self):
        # given
        sales_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
            StructField("sell_price", DoubleType(), True),
        ])
        sales_data = [
            ('STORE_1', 'ITEM_1', 1, 'd_7', 3, 1.5),
            ('STORE_2', 'ITEM_2', 2, 'd_8', 4, 3.5),
        ]
        sales = self.spark.createDataFrame(data=sales_data, schema=sales_schema)

        expected_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
            StructField("sell_price", DoubleType(), True),
            StructField("total_price", DoubleType(), True),
        ])
        expected_data = [
            ('STORE_1', 'ITEM_1', 1, 'd_7', 3, 1.5, 4.5),
            ('STORE_2', 'ITEM_2', 2, 'd_8', 4, 3.5, 14.0),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        # when
        actual = SalesProcessor.get_sales_with_total_prices(sales)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_top_n_stores_by_revenue(self):
        # given
        schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("week", IntegerType(), True),
            StructField("day", StringType(), True),
            StructField("unit_sales", IntegerType(), True),
            StructField("sell_price", DoubleType(), True),
            StructField("total_price", DoubleType(), True),
        ])
        data = [
            ('STORE_1', 'ITEM_1', 1, 'd_7', 2, 1.0, 2.0),
            ('STORE_1', 'ITEM_2', 1, 'd_7', 2, 2.0, 4.0),
            ('STORE_2', 'ITEM_3', 2, 'd_8', 1, 3.0, 3.0),
            ('STORE_3', 'ITEM_1', 2, 'd_8', 2, 4.0, 8.0),
        ]
        sales = self.spark.createDataFrame(data=data, schema=schema)

        expected_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("store_revenue", DoubleType(), True),
            StructField("store_rank", IntegerType(), True),
        ])
        expected_data = [
            ('STORE_3', 8.0, 1),
            ('STORE_1', 6.0, 2),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        n_stores_to_rank = 2
        # when
        actual = SalesProcessor.get_top_n_stores_by_revenue(n_stores_to_rank, sales)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_overwrite_csv_with_headers(self):
        # given
        output_path = path.join(self.test_dir, 'my_output_path')
        schema = StructType([
            StructField("foo", IntegerType(), True),
            StructField("bar", StringType(), True),
        ])
        expected = self.spark.createDataFrame(data=[(0, 'a')], schema=schema)
        # when
        SalesProcessor.overwrite_csv_with_headers(expected, output_path)
        # writes two times to overwrite
        SalesProcessor.overwrite_csv_with_headers(expected, output_path)
        actual = SalesProcessor.read_csv_with_headers(self.spark, output_path)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_sales_n_days_by_store_and_item(self):
        # given
        n_days_columns = ['d_1', 'd_2']
        schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("d_1", IntegerType(), True),
            StructField("d_2", IntegerType(), True),
        ])
        data = [
            ('STORE_1', 'ITEM_1', 5, 4),
            ('STORE_2', 'ITEM_2', 3, 2),
        ]
        sales = self.spark.createDataFrame(data=data, schema=schema)

        expected_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("sum_n_days_item_sales", IntegerType(), True),
        ])
        expected_data = [
            ('STORE_1', 'ITEM_1', 9),
            ('STORE_2', 'ITEM_2', 5),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        # when
        actual = SalesProcessor.get_sales_n_days_by_store_and_item(n_days_columns, sales)
        # then
        self.assert_dataframe_equals(actual, expected)

    def test_get_top_n_items_sold_by_store(self):
        # given
        schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("sum_n_days_item_sales", IntegerType(), True),
        ])
        data = [
            ('STORE_1', 'ITEM_1', 10),
            ('STORE_1', 'ITEM_2', 20),
            ('STORE_1', 'ITEM_3', 30),
            ('STORE_2', 'ITEM_4', 90),
            ('STORE_2', 'ITEM_5', 80),
            ('STORE_2', 'ITEM_6', 70),
        ]
        sales_by_store_and_item = self.spark.createDataFrame(data=data, schema=schema)

        expected_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("sum_n_days_item_sales", IntegerType(), True),
            StructField("item_rank", IntegerType(), True),
        ])
        expected_data = [
            ('STORE_1', 'ITEM_3', 30, 1),
            ('STORE_1', 'ITEM_2', 20, 2),
            ('STORE_2', 'ITEM_4', 90, 1),
            ('STORE_2', 'ITEM_5', 80, 2),
        ]
        expected = self.spark.createDataFrame(data=expected_data, schema=expected_schema)
        n_items_to_rank = 2
        # when
        actual = SalesProcessor.get_top_n_items_sold_by_store(n_items_to_rank, sales_by_store_and_item)
        # then
        self.assert_dataframe_equals(actual, expected)
