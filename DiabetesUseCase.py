# Databricks notebook source
# MAGIC %md #Daibetes Analytics
# MAGIC
# MAGIC Getting diabetes in middle age can reduce life span by about 10 years, claims a new research.
# MAGIC
# MAGIC A University of Oxford study of more than half a million people found those diagnosed with the condition before 50 lived an average of nine years less than those without the condition. That figure rose to ten years for patients in rural areas.
# MAGIC
# MAGIC Demonstrate the capabilities of Databricks as a Business Exploration, Advanced Protyping, Machine Learning and Production Refinement platform. 
# MAGIC
# MAGIC
# MAGIC ###1. Problem Statement
# MAGIC
# MAGIC Given a diabetes dataset availble from a publicly avialble dataset, we try to infer the different patterns that influence the outcome of diabetes. 
# MAGIC
# MAGIC ### 2. Experiment
# MAGIC   Can we extract imortant features that influnce the outcome of diabetes?
# MAGIC   
# MAGIC   Hypothesis: We can use data to predict which individual has a higher likelyhood of being suseptible to diabetes.
# MAGIC   
# MAGIC   For example, let's say we are able to extract features of individual demographic that can influence the outcome of the individual getting Diabetes. Using this information to help prevent other indivuals getting this by taking precaustionary measures.
# MAGIC   
# MAGIC   Can we create a predictive model that will predict the likelyhood whether an individual getting Diabetes?
# MAGIC
# MAGIC ### 3. Technical Solution
# MAGIC  
# MAGIC
# MAGIC - The datasets are as follows:
# MAGIC    - The contain the following user demograpics data like "pregnancies", "plasma glucose", "blood pressure", "triceps skin thickness", "insulin", "bmi", "diabetes pedigree", "age" 
# MAGIC - This data was extracted from a [Publicly available dataset](https://raw.githubusercontent.com/AvisekSukul/Regression_Diabetes/master/Custom%20Diabetes%20Dataset.csv)
# MAGIC  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Create** a cluster by...  
# MAGIC   - Click the `Clusters` icon on the left sidebar and then `Create Cluster.` 
# MAGIC   - Enter any text, i.e `demo` into the cluster name text box
# MAGIC   - Select the `Apache Spark Version` value `Spark 2.2 (auto-updating scala 2.11)`  
# MAGIC   - Click the `create cluster` button and wait for your cluster to be provisioned
# MAGIC   
# MAGIC **Attach** this notebook to your cluster by...   
# MAGIC   - Click on your cluster name in menu `Detached` at the top left of this workbook to attach it to this workbook 
# MAGIC   - Add the spark-xml library to the cluster created above. The library is present in libs folder under current user.

# COMMAND ----------

# DBTITLE 1,Step1: Ingest Data to Notebook
# MAGIC %md 
# MAGIC - Ingest data from dbfs sources 
# MAGIC - Create Sources as Tables
# MAGIC - If your running this notebook the first time, you would need to run the Setup_km notebook in the same folder.

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/diabetes.csv',header=True, sep=',', inferSchema=True)

# COMMAND ----------

# MAGIC %md display the data ingested to dataframe

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md Describe the schema of the dataset

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md The total number of records ingested

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step2: Explore the data
# MAGIC
# MAGIC - Review the Schema
# MAGIC - Profile the Data 
# MAGIC - Visualize the data

# COMMAND ----------

df.createOrReplaceTempView("diabetes")

# COMMAND ----------

# MAGIC %md Age-wise visualization of indiviuals in diabetes dataset

# COMMAND ----------

# DBTITLE 1,Visualize Data
# MAGIC %sql
# MAGIC select * from default.diabetes123 sort by age

# COMMAND ----------

# MAGIC %md Age-wise visualization of indiviuals with respect to Insulin level in diabetes dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from diabetes123 sort by age

# COMMAND ----------

# MAGIC %sql
# MAGIC select age, bmi from diabetes123 sort by age

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS diabetesrfm;
# MAGIC CREATE TABLE diabetesrfm as 
# MAGIC SELECT pregnancies, 
# MAGIC        'plasma glucose' as glucose,
# MAGIC        'blood pressure' as  bloodpressure,
# MAGIC        'triceps skin thickness' as skinthickness,
# MAGIC        insulin, bmi, 
# MAGIC        'diabetes pedigree' as pedigree, 
# MAGIC        age, diabetes
# MAGIC FROM default.diabetes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diabetesrfm LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC **Rename columns using PySpark**

# COMMAND ----------

sdf = spark.read.csv('/FileStore/tables/diabetes.csv', header=True, sep=',', inferSchema=True)

# COMMAND ----------

sdfrfm = sdf.selectExpr("pregnancies", "'plasma glucose' as glucose", "'blood pressure' as pressure", "'triceps skin thickness' as skinthickness", "insulin", "bmi", "'diabetes pedigree' as pedigree", "age", "diabetes")

sdfrfm.show(3)
sdfrfm.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC **A local Pandas DataFrame...**

# COMMAND ----------

import pandas as pd

Location = r"/dbfs/FileStore/tables/diabetes.csv"
pdf = pd.read_csv(Location)
pdf.columns

# COMMAND ----------

# MAGIC %md
# MAGIC Spark DataFrame...

# COMMAND ----------

display(sdfrfm)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark Dataframe shares some commands

# COMMAND ----------

sdfrfm.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark based way to sample view data 

# COMMAND ----------

sdf.show()    # works

pdf.show()        # doesn't work

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use the take method...

# COMMAND ----------

display(sdf.take(3))

# COMMAND ----------

# MAGIC %md
# MAGIC Describe the schema of the dataset

# COMMAND ----------

sdf.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC A couple of useful commands...

# COMMAND ----------

sdf.cache()     # Cache data for faster reuse

sdfnona = sdf.dropna()      # drop rows with missing values

# COMMAND ----------

# MAGIC %md
# MAGIC ### RDD Functions...

# COMMAND ----------

sdf.count()

# COMMAND ----------

sdfnona.count()

# COMMAND ----------

countsByAge = sdf.groupBy("age").count()
countsByAge.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Count of non null values by column...

# COMMAND ----------

pdf.count()

# COMMAND ----------

# DBTITLE 1,Spark Dataframe filter and sort
filterDF = sdf.filter(sdf.age > 20).sort('blood pressure')
display(filterDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Replacing Nulls

# COMMAND ----------

sdfclean = sdf.fillna("--")
display(sdfclean)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Explore the data
# MAGIC * Review the Schema  
# MAGIC * Profile the data   
# MAGIC * Visualize the data

# COMMAND ----------

# MAGIC %md
# MAGIC Create a temporary table from our spark dataframe...

# COMMAND ----------

sdf.createOrReplaceTempView("diabetesDF")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from diabetesDF limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC Saving a dataframe to a permanent managed table...

# COMMAND ----------

sdfrfm.write.saveAsTable("diabetsrfmDF")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from diabetsrfmDF limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize Data

# COMMAND ----------

# MAGIC %sql
# MAGIC select bmi, age from diabetesDF sort by age;

# COMMAND ----------

# MAGIC %md
# MAGIC Age-wise visualization of individuals with respect to insulin level in diabetes dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(diabetes) as freq, age from diabetesDF GROUP BY age;

# COMMAND ----------

# MAGIC %md
# MAGIC Misleading informations?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CASE 
# MAGIC         WHEN age is NULL THEN 'NA' 
# MAGIC         WHEN age < 30 then '< 30'
# MAGIC         WHEN age < 40 THEN '< 30 - 39'
# MAGIC         WHEN age < 50 THEN '< 40 - 49'
# MAGIC         WHEN age < 60 THEN '< 50 - 59'
# MAGIC         ELSE '60 and Over'
# MAGIC         END as AgeGrouping, 
# MAGIC SUM(diabetes) as diabetes, count(*) as frequency
# MAGIC FROM diabetes
# MAGIC GROUP BY AgeGrouping
# MAGIC SORT BY AgeGrouping;

# COMMAND ----------

# MAGIC %md
# MAGIC Use SQL to Create a Python Spark Dataframe...

# COMMAND ----------

dfsubset = spark.sql('SELECT * FROM databricks_ml.default.diabetes WHERE diabetes = 1')
display(dfsubset)

# COMMAND ----------

# MAGIC %md
# MAGIC Spark uses lazy execution so we need to force it to execute the query

# COMMAND ----------

plocaldf = (spark.sql('SELECT * FROM databricks_ml.default.diabetes WHERE diabetes = 1')).collect()
display(plocaldf)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert to local dataframe and use matplotlib...

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

plt.clf()
pdDF = sdf.toPandas()
pdDF.plot(x='age', y='pregnancies', kind='scatter', rot=45)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Adding a new column to a Spark Dataframe. Dataframes are immuable...

# COMMAND ----------

sparkDF = sdf.withColumn('Age2', sdf['age'].astype("string"))
sparkDF = sparkDF.drop('Age')
sparkDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC To add a column, you need to create a new dataframe...

# COMMAND ----------

pandasDF2 = pdf.copy()
pandasDF2['yearsoverminor'] = pdf['age'] - 16
pandasDF2.head()

# COMMAND ----------

sparkDF2 = sdf.withColumn('yearsoverminor', sdf['age'] - 16)
display(sparkDF2)

# COMMAND ----------

# MAGIC %md
# MAGIC Pushing work to the nodes with User Defined Functions (UDFs)...

# COMMAND ----------

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

yearssinceminor_udf = udf(lambda age: age-16)

# df = sqlContext.createDataFrame([{'name': 'Alice', 'age': 20}])
newsdf = sdf.withColumn('yearssinceminor', yearssinceminor_udf(sdf['age']))

# COMMAND ----------

newsdf.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Convert Spark Dataframe to local data frame...

# COMMAND ----------

pdDF = newsdf.toPandas()
pdDF

# COMMAND ----------

# MAGIC %md
# MAGIC Another want to convert a table to a Spark dataframe...

# COMMAND ----------

# Note Only numeric columns are included by default.

sdf2 = spark.table("databricks_ml.default.diabetes")
sdf2.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Use Databricks Display command to make the output more readable...

# COMMAND ----------

display(sdf2.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### PySpark SQL functions are native to the JVM on each node and so very performant

# COMMAND ----------



from pyspark.sql.functions import mean, min, max

sdf2.select([mean('age'), min('age'), max('age')]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Collect executes the query and returns the data to the head node.. Be careful the volume is small.

# COMMAND ----------

df.select('*').collect()

# COMMAND ----------



# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC library(psych)
# MAGIC
# MAGIC diabetesR <- collect(sample(sql("select * from diabetes"),withReplacement = FALSE, fraction = .1))
# MAGIC
# MAGIC # sample(withReplacement = FALSE, fraction = .1))
# MAGIC
# MAGIC pairs.panels(diabetesR)

# COMMAND ----------

# MAGIC %md Visualization of the Top 10 affected age groups with Diabetes

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from diabetes sort by age

# COMMAND ----------

# MAGIC %md ###Step3: Create the Features
# MAGIC
# MAGIC - Select features using SQL
# MAGIC - User Defined Functions to build custom features

# COMMAND ----------

# MAGIC %md
# MAGIC - Create an assemble `features` vector

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

train = VectorAssembler(
  inputCols = [
    "pregnancies", "plasma glucose", "blood pressure", "triceps skin thickness", "insulin", "bmi", 
    "diabetes pedigree", "age", "diabetes"
    ], 
  outputCol = "features").transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC - Display DataFrame: `AFTER PREPROCESSING`

# COMMAND ----------

display(train)

# COMMAND ----------

train1=train.withColumnRenamed("diabetes", "label")

# COMMAND ----------

# MAGIC %md  Schema of that feature vector dataframe

# COMMAND ----------

train1.printSchema()

# COMMAND ----------

# MAGIC %md ###Step3: Create the Model from the above feature Set
# MAGIC
# MAGIC - Select features using SQL
# MAGIC - User Defined Functions to build custom features

# COMMAND ----------

# MAGIC %md
# MAGIC Descision Tree Model

# COMMAND ----------

from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import *

# COMMAND ----------

dt = DecisionTreeClassifier()
eval = BinaryClassificationEvaluator(metricName ="areaUnderROC")
grid = ParamGridBuilder().baseOn(
  {
    dt.seed : 102993,
    dt.maxBins : 64
  }
).addGrid(
  dt.maxDepth, [4,6,8]
).build()

# COMMAND ----------

tvs = TrainValidationSplit(seed = 3923772, estimator=dt, trainRatio=0.7, evaluator = eval, estimatorParamMaps = grid)

# COMMAND ----------

model = tvs.fit(train1)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate model

# COMMAND ----------

model.validationMetrics

# COMMAND ----------

model.bestModel.write().overwrite().save("/models/dt")

# COMMAND ----------

# MAGIC %fs ls /models/

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.classification.DecisionTreeClassificationModel
# MAGIC
# MAGIC val dtModel = DecisionTreeClassificationModel.load("/models/dt")
# MAGIC display(dtModel)

# COMMAND ----------

# MAGIC %md
# MAGIC K- Means Model

# COMMAND ----------

from pyspark.ml.clustering import KMeans

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(train)

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate clustering

# COMMAND ----------

wssse = model.computeCost(train)
print("Within Set Sum of Squared Errors = " + str(wssse))

# COMMAND ----------

# MAGIC %md
# MAGIC Clustering Results

# COMMAND ----------

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

transformed = model.transform(train)

# COMMAND ----------

display(transformed)

# COMMAND ----------

transformed.printSchema()

# COMMAND ----------

# tdf = transformed.sample(False, fraction = 0.5)

# COMMAND ----------

display(
  transformed.groupBy("prediction").count()
)

# COMMAND ----------

display(
  transformed.groupBy("prediction").avg("insulin")
)

# COMMAND ----------

# DBTITLE 1,Results interpretation
# MAGIC
# MAGIC %md
# MAGIC
# MAGIC ### 1) Average `insulin` level for cluster with `prediction` = `0` is `32.21`. 
# MAGIC ### 2) Average `insulin` level for cluster with `prediction` = `1` is `253.71`.
# MAGIC
# MAGIC ![Diabetes-Analysis](https://img.huffingtonpost.com/asset/571e772b1900002e0056c26f.jpeg?cache=uhmcnbcaq7&ops=scalefit_720_noupscale)
# MAGIC
# MAGIC People with higher `insulin` level can be clubbed to people in cluster `#2` above. This increases the efficacy of predicting a diabetic patient.
# MAGIC
# MAGIC
