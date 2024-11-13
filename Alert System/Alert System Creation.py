from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when, count, lit, min as spark_min, max as spark_max
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType

spark = SparkSession.builder \
    .appName("PredictiveMaintenanceSystem") \
    .getOrCreate()

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

# Define UDFs for each check function
@udf(returnType=StringType())
def check_temperature(temperature, max_temperature):
    temp_pct = (temperature / max_temperature) * 100
    if temp_pct > 90:
        return "Danger"
    elif temp_pct > 80:
        return "Warning"
    return "Normal"


@udf(returnType=StringType())
def check_pressure(pressure, max_pressure):
    pressure_pct = (pressure / max_pressure) * 100
    if pressure_pct > 90:
        return "Danger"
    elif pressure_pct > 80:
        return "Warning"
    return "Normal"


@udf(returnType=StringType())
def check_rotational_speed(speed, max_speed):
    speed_pct = (speed / max_speed) * 100
    if speed_pct > 90:
        return "Danger"
    elif speed_pct > 80:
        return "Warning"
    return "Normal"

@udf(returnType=StringType())
def check_power_output(power, min_power, max_power):
    if min_power <= power <= 0.85 * max_power:
        return "Normal"
    elif 0.85 * max_power < power <= 0.95 * max_power:
        return "Warning"
    else:
        return "Danger"

@udf(returnType=StringType())
def check_noise_level(noise, min_noise, max_noise):
    warning_threshold = 85  # Fixed value based on industrial standards
    danger_threshold = 0.97 * max_noise

    if min_noise <= noise <= warning_threshold:
        return "Normal"
    elif warning_threshold < noise <= danger_threshold:
        return "Warning"
    else:
        return "Danger"

@udf(returnType=StringType())
def check_voltage(volt):
    nominal_voltage = 220
    normal_low = 0.85 * nominal_voltage  # -15% of nominal
    normal_high = 1.15 * nominal_voltage  # +15% of nominal
    warning_low = 0.80 * nominal_voltage  # -20% of nominal
    warning_high = 1.20 * nominal_voltage  # +20% of nominal

    if normal_low <= volt <= normal_high:
        return "Normal"
    elif warning_low <= volt < normal_low or normal_high < volt <= warning_high:
        return "Warning"
    else:
        return "Danger"

@udf(returnType=StringType())
def check_current(curr, min_curr, max_curr):
    normal_low = min_curr
    warning_threshold = 0.80 * max_curr
    danger_threshold = 0.90 * max_curr

    if normal_low <= curr <= warning_threshold:
        return "Normal"
    elif warning_threshold < curr <= danger_threshold:
        return "Warning"
    else:
        return "Danger"

@udf(returnType=StringType())
def check_oil_viscosity(viscosity):
    if viscosity < 25 or viscosity > 75:
        return "Danger"
    elif 25 <= viscosity < 30 or 70 < viscosity <= 75:
        return "Warning"
    else:
        return "Normal"

def add_alert_column(df):
    # Calculate min and max values
    stats = df.agg(
        spark_min("power_output").alias("min_power"),
        spark_max("power_output").alias("max_power"),
        spark_min("noise_level").alias("min_noise"),
        spark_max("noise_level").alias("max_noise"),
        spark_min("current").alias("min_current"),
        spark_max("current").alias("max_current")
    ).collect()[0]

    return df.withColumn("temperature_alert", check_temperature(col("temperature"), col("max_temperature"))) \
        .withColumn("pressure_alert", check_pressure(col("pressure"), col("max_pressure"))) \
        .withColumn("rotational_speed_alert", check_rotational_speed(col("rotational_speed"), col("max_rotational_speed"))) \
        .withColumn("power_output_alert", check_power_output(col("power_output"), lit(stats["min_power"]), lit(stats["max_power"]))) \
        .withColumn("noise_level_alert", check_noise_level(col("noise_level"), lit(stats["min_noise"]), lit(stats["max_noise"]))) \
        .withColumn("voltage_alert", check_voltage(col("voltage"))) \
        .withColumn("current_alert", check_current(col("current"), lit(stats["min_current"]), lit(stats["max_current"]))) \
        .withColumn("oil_viscosity_alert", check_oil_viscosity(col("oil_viscosity"))) \
        .withColumn("alert",
                    when((col("temperature_alert") == "Danger") |
                         (col("pressure_alert") == "Danger") |
                         (col("rotational_speed_alert") == "Danger") |
                         (col("power_output_alert") == "Danger") |
                         (col("noise_level_alert") == "Danger") |
                         (col("voltage_alert") == "Danger") |
                         (col("current_alert") == "Danger") |
                         (col("oil_viscosity_alert") == "Danger"), "Danger")
                    .when((col("temperature_alert") == "Warning") |
                          (col("pressure_alert") == "Warning") |
                          (col("rotational_speed_alert") == "Warning") |
                          (col("power_output_alert") == "Warning") |
                          (col("noise_level_alert") == "Warning") |
                          (col("voltage_alert") == "Warning") |
                          (col("current_alert") == "Warning") |
                          (col("oil_viscosity_alert") == "Warning"), "Warning")
                    .otherwise("Normal"))


def calculate_alert_percentages(df):
    total_count = df.count()

    alert_counts = df.groupBy("alert").agg(count("*").alias("count"))

    percentages = alert_counts.withColumn("percentage",
                                          (col("count") / lit(total_count) * 100).cast("decimal(5,2)"))

    return percentages


def calculate_detailed_alert_percentages(df):
    total_count = df.count()
    alert_columns = [
        "temperature_alert",
        "pressure_alert",
        "rotational_speed_alert",
        "power_output_alert",
        "noise_level_alert",
        "voltage_alert",
        "current_alert",
        "oil_viscosity_alert"
    ]

    detailed_percentages = {}

    for alert_col in alert_columns:
        counts = df.groupBy(alert_col).agg(count("*").alias("count"))
        percentages = counts.withColumn("percentage",
                                        (col("count") / lit(total_count) * 100).cast("decimal(5,2)"))
        detailed_percentages[alert_col] = percentages

    return detailed_percentages

# Main function to process the data
def process_data(data):
    # Add alert column
    data_with_alerts = add_alert_column(data)

    # Calculate alert percentages
    alert_percentages = calculate_alert_percentages(data_with_alerts)

    # Calculate detailed alert percentages
    detailed_alert_percentages = calculate_detailed_alert_percentages(data_with_alerts)

    return data_with_alerts, alert_percentages, detailed_alert_percentages


# Example usage
if __name__ == "__main__":
    # Load your data into a Spark DataFrame
    schema = define_schema()
    data = spark.read.option("header", "true").csv("Data Processing & Analysis/DEA & Feature Engineering/dataset_final/dataset_final.csv", schema=schema)

    # Process the data
    result_data, alert_percentages,  detailed_alert_percentages = process_data(data)

    # Save the data with the new alert column
    output_path = "Predictive-Maintenance-System-using-Apache-Spark/data_with_alerts"
    result_data.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

    # Show the results
    print("Sample of data with alerts:")
    result_data.select("equipment_id", "alert").show(10)

    print("\nAlert Percentages:")
    alert_percentages.show()

    print("\nDetailed Alert Percentages:")
    for alert_type, percentages in detailed_alert_percentages.items():
        print(f"\n{alert_type} Percentages:")
        percentages.show()

    # Stop the Spark session
    spark.stop()