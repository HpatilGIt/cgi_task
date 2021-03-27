# Sales processor

A Spark job to process a sales dataset

## Prerequisites

* PySpark 3.0.1+
* Pandas

## Running the tests
python -m unittest

## Job parameters

1. Sales dataset path where the following content exist:

<code>
.<br>
├── calendar.csv<br>
├── sales_train_validation.csv<br>
└── sell_prices.csv   
</code>
   
2. Output path where the following output folders are going to be created:

<code>
.<br>
├── top_n_items_sold_by_store<br>
└── top_n_stores_revenue<br>
</code>
