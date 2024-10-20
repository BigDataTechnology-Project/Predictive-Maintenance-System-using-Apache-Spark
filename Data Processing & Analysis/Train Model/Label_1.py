import pyspark
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType
from pyspark.sql import SparkSession

spark=SparkSession.builder.appName('Practise').getOrCreate()

def define_schema():
    return StructType([
        StructField("equipment_id", IntegerType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("vibration", DoubleType(), True),
        StructField("pressure", DoubleType(), True),
        StructField("rotational_speed", DoubleType(), True),
        StructField("power_output", DoubleType(), True),
        StructField("noise_level", DoubleType(), True),
        StructField("voltage", DoubleType(), True),
        StructField("current", DoubleType(), True),
        StructField("oil_viscosity", DoubleType(), True),
        StructField("model", StringType(), True),
        StructField("manufacturer", StringType(), True),
        StructField("installation_date", TimestampType(), True),
        StructField("max_temperature", DoubleType(), True),
        StructField("max_pressure", DoubleType(), True),
        StructField("max_rotational_speed", DoubleType(), True),
        StructField("expected_lifetime_years", DoubleType(), True),
        StructField("warranty_period_years", IntegerType(), True),
        StructField("last_major_overhaul", TimestampType(), True),
        StructField("location", StringType(), True),
        StructField("criticality", StringType(), True),
        StructField("maintenance_type", StringType(), True),
        StructField("description", StringType(), True),
        StructField("technician_id", IntegerType(), True),
        StructField("duration_hours", DoubleType(), True),
        StructField("cost", DoubleType(), True),
        StructField("parts_replaced", StringType(), True),
        StructField("maintenance_result", StringType(), True),
        StructField("maintenance_date", TimestampType(), True),
        StructField("production_rate", DoubleType(), True),
        StructField("operating_hours", DoubleType(), True),
        StructField("downtime_hours", DoubleType(), True),
        StructField("operator_id", IntegerType(), True),
        StructField("product_type", StringType(), True),
        StructField("raw_material_quality", StringType(), True),
        StructField("ambient_temperature", DoubleType(), True),
        StructField("ambient_humidity", DoubleType(), True),
        StructField("operation_date", TimestampType(), True),
        StructField("days_since_maintenance", IntegerType(), True),
        StructField("equipment_age_days", IntegerType(), True),
        StructField("days_since_overhaul", IntegerType(), True),
        StructField("temp_pct_of_max", DoubleType(), True),
        StructField("pressure_pct_of_max", DoubleType(), True),
        StructField("speed_pct_of_max", DoubleType(), True),
        StructField("cumulative_maintenance_cost", DoubleType(), True),
        StructField("cumulative_operating_hours", DoubleType(), True),
        StructField("estimated_rul", DoubleType(), True),
        StructField("criticality_encoded", DoubleType(), True),
        StructField("maintenance_type_encoded", DoubleType(), True),
        StructField("maintenance_result_encoded", DoubleType(), True),
        StructField("product_type_encoded", DoubleType(), True),
        StructField("raw_material_quality_encoded", DoubleType(), True),
        StructField("parts_replaced_encoded", DoubleType(), True)
    ])
schema = define_schema()
df_pyspark = spark.read.csv("C:\\Users\\admin\\Desktop\\University\\Big Data\\Predictive-Maintenance-System-using-Apache-Spark\\Data Processing & Analysis\\Dataset\\final_data_update.csv",header=True, schema = schema)
df_pyspark.printSchema()

df_pyspark.show(10)





##### Temperature
# Calculate the 60th and 90th percentiles for the temperature column
temperature_percentiles = df_pyspark.approxQuantile("temperature", [0.6, 0.9], 0.0)

# Extract the 60th and 90th percentile values
temperature_60th = temperature_percentiles[0]
temperature_90th = temperature_percentiles[1]

# Print the thresholds
print(f"Temperature 60th percentile (Normal to Warning boundary): {temperature_60th}")
print(f"Temperature 90th percentile (Warning to Danger boundary): {temperature_90th}")

from pyspark.sql.functions import when
# Create a new column 'temperature_category' based on the 60th and 90th percentile thresholds
df_pyspark = df_pyspark.withColumn(
    "temperature_category",
    when(df_pyspark["temperature"] <= temperature_60th, "Normal")
    .when((df_pyspark["temperature"] > temperature_60th) & (df_pyspark["temperature"] <= temperature_90th), "Warning")
    .otherwise("Danger")
)

# Show the result for temperature categories
df_pyspark.select("temperature", "temperature_category").show(10)

# Calculate total rows for temperature
total_rows_temperature = df_pyspark.count()

# Group by temperature_category and calculate counts
category_distribution_temperature = df_pyspark.groupBy("temperature_category").count()

# Calculate percentages for temperature
category_distribution_temperature = category_distribution_temperature.withColumn(
    "percentage", (category_distribution_temperature["count"] / total_rows_temperature) * 100
)

# Show the distribution for temperature categories
category_distribution_temperature.show()






##### Pressure
# Define a function to classify the pressure based on custom thresholds
def check_pressure(df, pressure_column="pressure"):
    df = df.withColumn(
        "pressure_category",
        when(df[pressure_column] <= 120, "Normal")
        .when((df[pressure_column] > 120) & (df[pressure_column] <= 140), "Warning")
        .otherwise("Danger")
    )
    return df

# Apply the function to classify pressure based on the custom thresholds
df_pyspark = check_pressure(df_pyspark)

# Show the result
df_pyspark.select("pressure", "pressure_category").show(10)

# Calculate total rows
total_rows = df_pyspark.count()

# Group by pressure_category and calculate counts
category_distribution = df_pyspark.groupBy("pressure_category").count()

# Calculate percentages
category_distribution = category_distribution.withColumn(
    "percentage", (category_distribution["count"] / total_rows) * 100
)

# Show the distribution
category_distribution.show()





##### Rotational speed
# Define a function to classify the rotational_speed based on custom thresholds
def check_rotational_speed(df, rotational_speed_column="rotational_speed"):
    df = df.withColumn(
        "rotational_speed_category",
        when(df[rotational_speed_column] <= 1100, "Normal")
        .when((df[rotational_speed_column] > 1100) & (df[rotational_speed_column] <= 1200), "Warning")
        .otherwise("Danger")
    )
    return df

# Apply the function to classify rotational_speed based on the custom thresholds
df_pyspark = check_rotational_speed(df_pyspark)

# Show the result
df_pyspark.select("rotational_speed", "rotational_speed_category").show(10)

# Calculate total rows for rotational speed
total_rows_rotational_speed = df_pyspark.count()

# Group by rotational_speed_category and calculate counts
category_distribution_rotational_speed = df_pyspark.groupBy("rotational_speed_category").count()

# Calculate percentages for rotational speed
category_distribution_rotational_speed = category_distribution_rotational_speed.withColumn(
    "percentage", (category_distribution_rotational_speed["count"] / total_rows_rotational_speed) * 100
)

# Show the distribution for rotational speed categories
category_distribution_rotational_speed.show()







##### Noise level
# Define a function to classify the noise_level based on custom thresholds
def check_noise_level(df, noise_level_column="noise_level"):
    df = df.withColumn(
        "noise_level_category",
        when(df[noise_level_column] <= 75, "Normal")
        .when((df[noise_level_column] > 75) & (df[noise_level_column] <= 82), "Warning")
        .otherwise("Danger")
    )
    return df

# Apply the function to classify noise_level based on the custom thresholds
df_pyspark = check_noise_level(df_pyspark)

# Show the result
df_pyspark.select("noise_level", "noise_level_category").show(10)



# Calculate total rows for noise level
total_rows_noise_level = df_pyspark.count()

# Group by noise_level_category and calculate counts
category_distribution_noise_level = df_pyspark.groupBy("noise_level_category").count()

# Calculate percentages for noise level
category_distribution_noise_level = category_distribution_noise_level.withColumn(
    "percentage", (category_distribution_noise_level["count"] / total_rows_noise_level) * 100
)

# Show the distribution for noise level categories
category_distribution_noise_level.show()



#### Voltage
# Define a function to classify the voltage based on custom thresholds
def check_voltage(df, voltage_column="voltage"):
    df = df.withColumn(
        "voltage_category",
        when(df[voltage_column] <= 225, "Normal")
        .when((df[voltage_column] > 225) & (df[voltage_column] <= 237), "Warning")
        .otherwise("Danger")
    )
    return df

# Apply the function to classify voltage based on the custom thresholds
df_pyspark = check_voltage(df_pyspark)

# Show the result
df_pyspark.select("voltage", "voltage_category").show(10)


# Calculate total rows for voltage
total_rows_voltage = df_pyspark.count()

# Group by voltage_category and calculate counts
category_distribution_voltage = df_pyspark.groupBy("voltage_category").count()

# Calculate percentages for voltage
category_distribution_voltage = category_distribution_voltage.withColumn(
    "percentage", (category_distribution_voltage["count"] / total_rows_voltage) * 100
)

# Show the distribution for voltage categories
category_distribution_voltage.show()






##### Combine the condition together
from pyspark.sql.functions import when

# Combine all the conditions into a single "system_warning" column, including the temperature_category
df_pyspark = df_pyspark.withColumn(
    "system_warning",
    when(
        (df_pyspark["pressure_category"] == "Danger") |
        (df_pyspark["rotational_speed_category"] == "Danger") |
        (df_pyspark["noise_level_category"] == "Danger") |
        (df_pyspark["voltage_category"] == "Danger") |
        (df_pyspark["temperature_category"] == "Danger"), "Danger"
    ).when(
        (df_pyspark["pressure_category"] == "Warning") |
        (df_pyspark["rotational_speed_category"] == "Warning") |
        (df_pyspark["noise_level_category"] == "Warning") |
        (df_pyspark["voltage_category"] == "Warning") |
        (df_pyspark["temperature_category"] == "Warning"), "Warning"
    ).otherwise("Normal")
)

# Drop the individual category columns if not needed
df_pyspark = df_pyspark.drop("pressure_category", "rotational_speed_category", "noise_level_category", "voltage_category", "temperature_category")

# Show the result with just the system warning
df_pyspark.select("system_warning").show(10)




##### Train model
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Step 1: Split the dataset into training and test sets (80% train, 20% test)
train_data, test_data = df_pyspark.randomSplit([0.8, 0.2], seed=42)

# Step 2: Convert the 'system_warning' label to numeric using StringIndexer
indexer = StringIndexer(inputCol="system_warning", outputCol="label")
train_data = indexer.fit(train_data).transform(train_data)
test_data = indexer.fit(test_data).transform(test_data)

# Step 3: Assemble features (you can choose any relevant features for your model)
assembler = VectorAssembler(
    inputCols=[
        "temperature", "rotational_speed", "noise_level", "voltage",
        "pressure"  # Add other features if needed
    ],
    outputCol="features"
)

train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

# Step 4: Train the model (RandomForestClassifier in this case)
rf = RandomForestClassifier(featuresCol="features", labelCol="label")
model = rf.fit(train_data)

# Step 5: Test the model on the test set
predictions = model.transform(test_data)

# Step 6: Evaluate the model using accuracy metric
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# Step 7: Show some of the predictions
predictions.select("features", "system_warning", "prediction").show(10)


# Check distinct values of system_warning and their corresponding labels
predictions.select("system_warning", "label").distinct().show()


# Calculate the total number of rows in the dataset
total_rows = train_data.count()

# Group by 'system_warning' and calculate the counts
category_distribution = train_data.groupBy("system_warning").count()

# Calculate the percentages for each category
category_distribution = category_distribution.withColumn(
    "percentage", (category_distribution["count"] / total_rows) * 100
)

# Show the result with both count and percentage
category_distribution.show()

