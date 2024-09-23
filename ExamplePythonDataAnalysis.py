# Databricks notebook source
# MAGIC %md # PySpark Data Analysis Basics 
# MAGIC
# MAGIC ![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) ![Spark Logo](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png)
# MAGIC ### This notebook was started with the notebook on GitHub by Todd Lisonbee. Link below.
# MAGIC https://github.com/trustedanalytics/jupyter-default-notebooks/blob/master/notebooks/examples/spark/pyspark-dataframe-basics.ipynb
# MAGIC
# MAGIC This notebook demonstrates some basic DataFrame operations in PySpark.
# MAGIC
# MAGIC Several [Spark examples](https://raw.githubusercontent.com/tree/examples/spark) are included with TAP.
# MAGIC
# MAGIC More examples are available on the Spark website: http://spark.apache.org/examples.html
# MAGIC
# MAGIC PySpark API documentation: http://spark.apache.org/docs/latest/api/python/

# COMMAND ----------

# MAGIC %md  ## When to use RDDs over Data Frames...
# MAGIC
# MAGIC There are three main ways data is represented as a Spark object which are resilient distributed datasets, data frames, and datasets.  
# MAGIC
# MAGIC Generally, dataframes provide ease of use and the best performance.  Python (PySPark) does not currently support datasets.  RDDs are the original way Spark exposed data and may make sense in some use cases.
# MAGIC See the link below for more information on when to use each. 
# MAGIC
# MAGIC https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html

# COMMAND ----------

# MAGIC %md  ### With Databricks, there's no need to import PySpark or create a Spark Context...

# COMMAND ----------

data = [
    (1, 'a'), 
    (2, 'b'), 
    (3, 'c'), 
    (4, 'd'), 
    (5, 'e'), 
    (6, 'a'), 
    (7, 'b'), 
    (8, 'c'), 
    (9, 'd'), 
    (10, 'e')
]

# Convert a local data set into a DataFrame
df = sqlContext.createDataFrame(data, ['numbers', 'letters'])

display(df)

# Convert to a Pandas DataFrame for easy display.  Not needed with Databricks.  
df.toPandas()  

PDF = df.toPandas # Must assign to an object for the pandas data frame to be saved.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Sometimes you need to convert a Spark data frame to a Pandas data frame to do some work on it like displaying it.
# MAGIC #### Azure Databricks includes the display function to allow you to display data from the Spark dataframe without the need to convert it to Pandas.

# COMMAND ----------

# Count the number of rows in the DataFrame
print(df.count())

# COMMAND ----------

# DBTITLE 1,View some rows in the default Spark manner...
print(df.take(3))  # take is a Spark data frame method

# COMMAND ----------

# DBTITLE 1,Show the rows in the Databricks way...
display(df.take(3))

# COMMAND ----------

# Sort descending
descendingDf = df.orderBy(df.numbers.desc())

# View some rows.  Wihtout the display function, you would need to convert the dataframe to pandas before displaying it.
display(descendingDf.take(5))

# COMMAND ----------

# DBTITLE 1,Be Sure to Filter Data Before Converting It to a pandas Data Frame...
# Filter the DataFrame
filtered = df.where(df.numbers < 5)

# Convert to Pandas DataFrame for easy viewing
filtered.toPandas()

# COMMAND ----------



# COMMAND ----------

# import some more functions
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import avg
from pyspark.sql.functions import sum

# Perform aggregations on the DataFrame
agg = df.agg(
    avg(df.numbers).alias("avg_numbers"), 
    sum(df.numbers).alias("sum_numbers"),
    countDistinct(df.numbers).alias("distinct_numbers"), 
    countDistinct(df.letters).alias('distinct_letters')
)

# Convert the results to Pandas DataFrame
agg.toPandas()

# COMMAND ----------

# View some summary statistics
df.describe().show()

# COMMAND ----------

# MAGIC %md ## Stop the Spark Context

# COMMAND ----------

# Stop the context when you are done with it. When you stop the SparkContext resources 
# are released and no further operations can be performed within that context
sc.stop()

# COMMAND ----------

# DBTITLE 1,File is available via Databricks...
# Use the Spark CSV datasource with options specifying:
# - First line of file is a header
# - Automatically infer the schema of the data
data = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("/databricks-datasets/samples/population-vs-price/data_geo.csv")

data2 = data.toDF('2014_Rank', 'City', 'State', 'StateCode','2014PopulationEst','2015MedianSalesPrice')

data2.cache() # Cache data for faster reuse
data2 = data2.dropna() # drop rows with missing values

# Register table so it is accessible via SQL Context
# For Apache Spark = 2.0
data2.createOrReplaceTempView("data_geo")

# COMMAND ----------

# MAGIC %sql select 2014_Rank, city, statecode from data_geo limit 5;

# COMMAND ----------

geoSPF = spark.sql("select * from data_geo limit 5")
geoSPF.take(5)
display(geoSPF)

# COMMAND ----------

# DBTITLE 1,The CSV file was uploaded using the Data table create GUI but cancelled without creating a table...
# Use the Spark CSV datasource with options specifying:
# - First line of file is a header
# - Automatically infer the schema of the data
sales = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/mydata/sales.csv")

sales.cache() # Cache data for faster reuse
sales = sales.dropna() # drop rows with missing values

# Register table so it is accessible via SQL Context
# For Apache Spark = 2.0
sales.createOrReplaceTempView("sales_view")

# COMMAND ----------

# MAGIC %sql select * from sales_view limit 5;

# COMMAND ----------

birthwtDF = spark.sql("select * from birthwt_view limit 5")

# COMMAND ----------

display(birthwtDF)
