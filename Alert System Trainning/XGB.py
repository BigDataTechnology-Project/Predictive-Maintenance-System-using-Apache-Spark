from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType
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

# Load the data with alerts
data = spark.read.option("header", "true").csv("data_with_alert.csv")

# Define feature columns and their types
feature_cols = [
    "temperature", "vibration", "pressure", "rotational_speed", "power_output",
    "noise_level", "voltage", "current", "oil_viscosity", "production_rate",
    "operating_hours", "downtime_hours", "ambient_temperature", "ambient_humidity",
    "days_since_maintenance", "equipment_age_days", "days_since_overhaul",
    "temp_pct_of_max", "pressure_pct_of_max", "speed_pct_of_max",
    "cumulative_maintenance_cost", "cumulative_operating_hours", "estimated_rul",
    "criticality_encoded", "maintenance_type_encoded", "maintenance_result_encoded",
    "product_type_encoded", "raw_material_quality_encoded"
]

# Cast string columns to appropriate numeric types
for column in feature_cols:
    if column in ["days_since_maintenance", "equipment_age_days", "days_since_overhaul"]:
        data = data.withColumn(column, col(column).cast(IntegerType()))
    else:
        data = data.withColumn(column, col(column).cast(DoubleType()))
# Create feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Create label index
label_indexer = StringIndexer(inputCol="alert", outputCol="label")
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
print("\nTop 10 Important Features:")
for feature, score in sorted_importance[:10]:
    print(f"{feature}: {score}")

# Save the model
model.save_model('xgboost_alert_model.json')
print("\nModel saved as 'xgboost_alert_model.json'")

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