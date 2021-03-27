import time
from sys import argv

from pyspark.sql import SparkSession

from com.cgi.sales.sales_processor import SalesProcessor


def run_sales_processor():
    """
    Function in charge of running the sales processor.
    """
    start_time = time.time()
    processor = SalesProcessor(spark, dataset_path, output_path)
    processor.run()
    end_time = time.time()
    print('Job total runtime: ', str(int(end_time - start_time)) + 's')
    spark.stop()


def create_spark_session():
    """
    Function to create an Spark session.
    :return: a Spark session.
    """
    return SparkSession \
        .builder \
        .appName("Sales processor") \
        .master('local[*]') \
        .getOrCreate()


if __name__ == '__main__':
    """
    Spark application entry point.
    """
    _, dataset_path, output_path = argv
    print('Creating a spark session.')
    spark = create_spark_session()
    print('Spark session created.')
    print('Running the sales processor.')
    run_sales_processor()
    print('Sales processor finished.')
