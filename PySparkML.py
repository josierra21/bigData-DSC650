from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import happybase
from datetime import datetime

# Step 1: Create a Spark session (Hive support enabled)
spark = SparkSession.builder \
    .appName("Water Potability MLlib Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# Step 2: Load data from the Hive table into a Spark DataFrame
water_df = spark.sql("""
    SELECT
        ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,
        Organic_carbon, Trihalomethanes, Turbidity,
        Potability
    FROM water_quality
""")

# Step 3: Handle null values
water_df = water_df.na.drop()

# Step 4: Assemble features into a single vector column
feature_cols = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

assembled_df = assembler.transform(water_df) \
    .select("features", water_df["Potability"].alias("label"))

# Step 5: Split into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Step 6: Train Logistic Regression model (binary classification)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
model = lr.fit(train_data)

# Step 7: Make predictions
predictions = model.transform(test_data)

# Step 8: Evaluate model (AUC + Accuracy)
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = auc_evaluator.evaluate(predictions)

acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = acc_evaluator.evaluate(predictions)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")

# Write metrics to HBase with happybase 
# Table: water_metrics, Column Family: cf
# Row key: unique run id so each run is stored separately
run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

data = [
    (run_id, "cf:auc", str(auc)),
    (run_id, "cf:accuracy", str(accuracy))
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection("master")
    connection.open()
    table = connection.table("water_metrics")
    for row_key, column, value in partition:
        table.put(row_key, {column: value})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Stop Spark session
spark.stop()