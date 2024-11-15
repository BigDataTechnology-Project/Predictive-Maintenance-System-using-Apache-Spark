from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType

def create_spark_session():
    return (SparkSession.builder
            .appName("PredictiveMaintenanceFeatureEngineering")
            .getOrCreate())

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
        StructField("days_since_maintenance", IntegerType(), True)
    ])

def feature_engineering(df):
    # Calculate equipment age
    df = df.withColumn("equipment_age_days",
                       F.datediff(df.timestamp, df.installation_date))

    # Calculate days since overhaul (absolute value)
    df = df.withColumn("days_since_overhaul",
                       F.abs(F.datediff(df.timestamp, df.last_major_overhaul)))

    # Calculate percentage of max for temperature, pressure, and speed
    df = df.withColumn("temp_pct_of_max", df.temperature / df.max_temperature * 100)
    df = df.withColumn("pressure_pct_of_max", df.pressure / df.max_pressure * 100)
    df = df.withColumn("speed_pct_of_max", df.rotational_speed / df.max_rotational_speed * 100)

    # Calculate cumulative maintenance cost and operating hours
    window_cumulative = Window.partitionBy("equipment_id").orderBy("timestamp").rangeBetween(Window.unboundedPreceding, 0)
    df = df.withColumn("cumulative_maintenance_cost", F.sum("cost").over(window_cumulative))
    df = df.withColumn("cumulative_operating_hours", F.sum("operating_hours").over(window_cumulative))

    # Estimate Remaining Useful Life (RUL) - This is a placeholder calculation
    df = df.withColumn("estimated_rul",
                       df.expected_lifetime_years * 365 - df.equipment_age_days)

    # Take absolute value of days_since_maintenance
    df = df.withColumn("days_since_maintenance", F.abs(df.days_since_maintenance))

    return df

def select_columns(df):
    return df.select("*")


def label_encoding(df):
    categorical_cols = ["criticality", "maintenance_type", "maintenance_result",
                        "product_type", "raw_material_quality", "parts_replaced"]

    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_encoded", handleInvalid="keep")
        model = indexer.fit(df)
        df = model.transform(df)

        # Print the mapping for this column
        print(f"\nEncoding mapping for {col}:")
        for idx, category in enumerate(model.labels):
            print(f"{idx}: {category}")

    return df


def validate_equipment_age(df):
    # Check for negative equipment age
    negative_age = df.filter(F.col("equipment_age_days") < 0)

    # Get count and basic stats
    neg_count = negative_age.count()

    print(f"Number of records with negative equipment age: {neg_count}")

    if neg_count > 0:
        print("\nSample of records with negative age:")
        negative_age.select(
            "equipment_id",
            "timestamp",
            "installation_date",
            "equipment_age_days"
        ).orderBy("equipment_age_days").show(5)

        # Get statistics for negative ages
        negative_age.select(
            F.min("equipment_age_days").alias("min_negative_age"),
            F.max("equipment_age_days").alias("max_negative_age"),
            F.avg("equipment_age_days").alias("avg_negative_age")
        ).show()

    return negative_age


def analyze_invalid_timestamps(df):
    # Đếm số lượng bản ghi có timestamp < installation_date
    invalid_records = df.filter(F.col("timestamp") < F.col("installation_date"))
    total_records = df.count()
    invalid_count = invalid_records.count()

    print(f"Total records: {total_records}")
    print(f"Records with timestamp before installation_date: {invalid_count}")
    print(f"Percentage: {(invalid_count / total_records * 100):.2f}%")

    # Hiển thị phân bố theo equipment_id
    print("\nDistribution by equipment_id:")
    invalid_records.groupBy("equipment_id") \
        .count() \
        .orderBy("equipment_id") \
        .show(5)

    # Hiển thị mẫu chi tiết
    print("\nSample records with invalid timestamps:")
    invalid_records.select(
        "equipment_id",
        "timestamp",
        "installation_date",
        "equipment_age_days"
    ).orderBy("equipment_id", "timestamp") \
        .show(5)


def main():
    spark = create_spark_session()

    # Load your integrated dataset
    schema = define_schema()
    df = spark.read.csv("Data Processing & Analysis/DEA & Feature Engineering/cleaned_dataset.csv/cleaned_dataset.csv", header=True,
                        schema=schema)

    # Perform feature engineering
    df_engineered = feature_engineering(df)

    # Select necessary columns
    df_selected = select_columns(df_engineered)

    # Perform label encoding and drop original categorical columns
    df_encoded = label_encoding(df_selected)

    # Order the DataFrame by equipment_id
    df_ordered = df_encoded.orderBy("equipment_id", "timestamp")

    # Print schema after all transformations
    print("\nFinal schema:")
    df_ordered.printSchema()

    # Save the resulting dataframe
    df_ordered.coalesce(1).write.mode("overwrite").option("header", "true").csv("Data Processing & Analysis/DEA & Feature Engineering/dataset_final")

    spark.stop()

if __name__ == "__main__":
    main()
