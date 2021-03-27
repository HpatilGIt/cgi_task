import logging
import unittest

from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession, DataFrame


class PySparkTest(unittest.TestCase):
    spark: SparkSession = None

    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession.builder
                .master('local[2]')
                .appName('Testing pyspark session')
                .enableHiveSupport()
                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    @staticmethod
    def assert_dataframe_equals(actual: DataFrame, expected: DataFrame):
        """
        Function to compare two Spark dataframes using Pandas
        """
        key_columns = expected.columns
        actual_sorted = actual.toPandas().sort_values(by=key_columns).reset_index(drop=True)
        expected_sorted = expected.toPandas().sort_values(by=key_columns).reset_index(drop=True)
        assert_frame_equal(actual_sorted, expected_sorted)
