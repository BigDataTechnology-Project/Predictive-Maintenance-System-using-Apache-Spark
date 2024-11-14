from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType, StringType, TimestampType
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import time

# Initialize Spark session
spark = SparkSession.builder \
       .appName("XGBoostAlertPrediction") \
       .config("spark.driver.memory", "8g") \
       .config("spark.executor.memory", "8g") \
       .config("spark.driver.maxResultSize", "4g") \
       .getOrCreate()

def define_schema():
    return StructType([
        StructField("equipment_id", IntegerType(), nullable=True),
        StructField("timestamp", TimestampType(), nullable=True),
        StructField("temperature", DoubleType(), nullable=True),
        StructField("vibration", DoubleType(), nullable=True),
        StructField("pressure", DoubleType(), nullable=True),
        StructField("rotational_speed", DoubleType(), nullable=True),
        StructField("power_output", DoubleType(), nullable=True),
        StructField("noise_level", DoubleType(), nullable=True),
        StructField("voltage", DoubleType(), nullable=True),
        StructField("current", DoubleType(), nullable=True),
        StructField("oil_viscosity", DoubleType(), nullable=True),
        StructField("model", StringType(), nullable=True),
        StructField("manufacturer", StringType(), nullable=True),
        StructField("installation_date", TimestampType(), nullable=True),
        StructField("max_temperature", DoubleType(), nullable=True),
        StructField("max_pressure", DoubleType(), nullable=True),
        StructField("max_rotational_speed", DoubleType(), nullable=True),
        StructField("expected_lifetime_years", DoubleType(), nullable=True),
        StructField("warranty_period_years", IntegerType(), nullable=True),
        StructField("last_major_overhaul", TimestampType(), nullable=True),
        StructField("location", StringType(), nullable=True),
        StructField("criticality", StringType(), nullable=True),
        StructField("maintenance_type", StringType(), nullable=True),
        StructField("description", StringType(), nullable=True),
        StructField("technician_id", IntegerType(), nullable=True),
        StructField("duration_hours", DoubleType(), nullable=True),
        StructField("cost", DoubleType(), nullable=True),
        StructField("parts_replaced", StringType(), nullable=True),
        StructField("maintenance_result", StringType(), nullable=True),
        StructField("maintenance_date", TimestampType(), nullable=True),
        StructField("production_rate", DoubleType(), nullable=True),
        StructField("operating_hours", DoubleType(), nullable=True),
        StructField("downtime_hours", DoubleType(), nullable=True),
        StructField("operator_id", IntegerType(), nullable=True),
        StructField("product_type", StringType(), nullable=True),
        StructField("raw_material_quality", StringType(), nullable=True),
        StructField("ambient_temperature", DoubleType(), nullable=True),
        StructField("ambient_humidity", DoubleType(), nullable=True),
        StructField("operation_date", TimestampType(), nullable=True),
        StructField("days_since_maintenance", IntegerType(), nullable=True),
        StructField("equipment_age_days", IntegerType(), nullable=True),
        StructField("days_since_overhaul", IntegerType(), nullable=True),
        StructField("temp_pct_of_max", DoubleType(), nullable=True),
        StructField("pressure_pct_of_max", DoubleType(), nullable=True),
        StructField("speed_pct_of_max", DoubleType(), nullable=True),
        StructField("cumulative_maintenance_cost", DoubleType(), nullable=True),
        StructField("cumulative_operating_hours", DoubleType(), nullable=True),
        StructField("estimated_rul", DoubleType(), nullable=True),
        StructField("criticality_encoded", DoubleType(), nullable=True),
        StructField("maintenance_type_encoded", DoubleType(), nullable=True),
        StructField("maintenance_result_encoded", DoubleType(), nullable=True),
        StructField("product_type_encoded", DoubleType(), nullable=True),
        StructField("raw_material_quality_encoded", DoubleType(), nullable=True),
        StructField("parts_replaced_encoded", DoubleType(), nullable=True),
        StructField("temperature_alert", StringType(), nullable=True),
        StructField("pressure_alert", StringType(), nullable=True),
        StructField("rotational_speed_alert", StringType(), nullable=True),
        StructField("power_output_alert", StringType(), nullable=True),
        StructField("noise_level_alert", StringType(), nullable=True),
        StructField("voltage_alert", StringType(), nullable=True),
        StructField("current_alert", StringType(), nullable=True),
        StructField("oil_viscosity_alert", StringType(), nullable=True),
        StructField("alert", StringType(), nullable=True),
        StructField("alert_score", IntegerType(), nullable=True),
        StructField("age_condition_score", IntegerType(), nullable=False),
        StructField("downtime_condition_score", IntegerType(), nullable=False),
        StructField("maintenance_condition_score", IntegerType(), nullable=False),
        StructField("environment_condition_score", IntegerType(), nullable=False),
        StructField("criticality_avg_annual_cost", DoubleType(), nullable=True),
        StructField("threshold", DoubleType(), nullable=True),
        StructField("maintenance_cost_condition_score", IntegerType(), nullable=False),
        StructField("operational_score", IntegerType(), nullable=False),
        StructField("total_score", IntegerType(), nullable=True),
        StructField("maintenance_needed", StringType(), nullable=False)
    ])


# Load the data with alerts
schema = define_schema()
data = spark.read.csv("Maintenance Recommendation System/df_maintenance.csv",header=True,schema=schema)

# Define feature columns and their types
feature_cols = ["temperature", "vibration", "pressure", "rotational_speed", "power_output",
    "noise_level", "voltage", "current", "oil_viscosity", "production_rate",
    "operating_hours", "downtime_hours", "ambient_temperature", "ambient_humidity",
    "days_since_maintenance", "equipment_age_days", "expected_lifetime_years",
    "warranty_period_years", "cumulative_maintenance_cost"]

# Create feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Create label index
label_indexer = StringIndexer(inputCol="maintenance_needed", outputCol="label")
data = label_indexer.fit(data).transform(data)

# Convert Spark DataFrame to pandas DataFrame
pandas_df = data.select("features", "label").toPandas()

# Extract features and labels
X = np.array(pandas_df['features'].tolist())
y = pandas_df['label'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'multi:softprob',
    'num_class': 3  # 3 classes: Normal, Warning, Danger
}

# Train the model
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Make predictions
preds = model.predict(dtest)
best_preds = np.argmax(preds, axis=1)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, best_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_preds))

# Feature importance
importance = model.get_score(importance_type='weight')
feature_importance = [(feature_cols[int(feature.replace('f', ''))], score) for feature, score in importance.items()]
sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
print("\nTop 5 Important Features:")
for feature, score in sorted_importance[:5]:
    print(f"{feature}: {score}")
# Stop Spark session
spark.stop()

print("Process completed!")

# Example prediction
print("\nExample Prediction:")
# Choose a random sample from the test set
random_index = np.random.randint(0, len(X_test))
sample = X_test[random_index]
true_label = y_test[random_index]

# Make prediction
sample_dmatrix = xgb.DMatrix([sample])
prediction = model.predict(sample_dmatrix)[0]
predicted_class = np.argmax(prediction)

print(f"True label: {true_label}")
print(f"Predicted probabilities: {prediction}")
print(f"Predicted class: {predicted_class}")

# Print feature values for this sample
print("\nFeature values for this sample:")
for feature, value in zip(feature_cols, sample):
    print(f"{feature}: {value:.2f}")
