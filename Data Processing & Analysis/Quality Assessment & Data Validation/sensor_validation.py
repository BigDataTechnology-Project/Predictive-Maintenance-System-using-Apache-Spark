from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, max as spark_max, count, lag, datediff, hour, lit, broadcast
from pyspark.sql.window import Window
import pyspark.sql.functions as F


def create_spark_session():
    return (SparkSession.builder
            .appName("SensorDataValidation")
            .master("local[*]")
            .config("spark.driver.memory", "8g")  # Increased from 4g to 8g
            .config("spark.executor.memory", "8g")  # Increased from 4g to 8g
            .config("spark.python.worker.memory", "6g")  # Increased from 4g to 6g
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", "100")
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.6")
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
            .config("spark.sql.autoBroadcastJoinThreshold", "10m")
            .config("spark.sql.broadcastTimeout", "300")
            .getOrCreate())


def verify_timestamp_range_and_continuity(sensor_df):
    # Get timestamp range
    time_range = sensor_df.agg(
        F.min("timestamp").alias("min_timestamp"),
        F.max("timestamp").alias("max_timestamp")
    ).collect()[0]

    # Check for gaps in timestamps
    window_spec = Window.partitionBy("equipment_id").orderBy("timestamp")
    gaps_df = sensor_df.withColumn("prev_timestamp", F.lag("timestamp").over(window_spec))
    gaps_df = gaps_df.withColumn("time_diff", F.when(col("prev_timestamp").isNotNull(),
                                                     F.unix_timestamp("timestamp") - F.unix_timestamp("prev_timestamp"))
                                 .otherwise(lit(0)))
    max_gap = gaps_df.agg(F.max("time_diff").alias("max_gap_seconds")).collect()[0]["max_gap_seconds"]

    return {
        "min_timestamp": time_range["min_timestamp"],
        "max_timestamp": time_range["max_timestamp"],
        "max_gap_seconds": max_gap
    }


def check_reading_ranges(sensor_df):
    numeric_columns = ["temperature", "vibration", "pressure", "rotational_speed",
                       "power_output", "noise_level", "voltage", "current", "oil_viscosity"]

    range_checks = {}
    for column in numeric_columns:
        stats = sensor_df.agg(
            F.min(column).alias(f"min_{column}"),
            F.max(column).alias(f"max_{column}"),
            F.count(F.when(col(column) < 0, True)).alias(f"negative_count_{column}")
        ).collect()[0]

        range_checks[column] = {
            "min": stats[f"min_{column}"],
            "max": stats[f"max_{column}"],
            "negative_count": stats[f"negative_count_{column}"]
        }

    return range_checks


def analyze_reading_frequency(sensor_df):
    frequency_df = sensor_df.groupBy("equipment_id") \
        .agg(F.count("*").alias("reading_count"),
             F.min("timestamp").alias("first_reading"),
             F.max("timestamp").alias("last_reading"))

    frequency_df = frequency_df.withColumn("time_span_hours",
                                           F.round((F.unix_timestamp("last_reading") - F.unix_timestamp(
                                               "first_reading")) / 3600, 2))

    frequency_df = frequency_df.withColumn("avg_readings_per_hour",
                                           F.round(col("reading_count") / col("time_span_hours"), 2))

    return frequency_df.collect()


def verify_equipment_ids(spark, sensor_df, equipment_df):
    # Broadcast the equipment dataframe as it's likely smaller
    equipment_df_broadcast = broadcast(equipment_df)

    # Get distinct equipment IDs from sensor data
    sensor_equipment_ids = sensor_df.select("equipment_id").distinct()

    # Perform a left anti join to find missing IDs
    missing_ids = sensor_equipment_ids.join(
        equipment_df_broadcast,
        sensor_equipment_ids.equipment_id == equipment_df_broadcast.equipment_id,
        "left_anti"
    )

    # Perform a left anti join to find extra IDs
    extra_ids = equipment_df_broadcast.join(
        sensor_equipment_ids,
        sensor_equipment_ids.equipment_id == equipment_df_broadcast.equipment_id,
        "left_anti"
    )

    # Count the missing and extra IDs
    missing_count = missing_ids.count()
    extra_count = extra_ids.count()

    # Only collect the IDs if there are mismatches (to avoid unnecessary data transfer)
    missing_id_list = missing_ids.rdd.map(lambda x: x.equipment_id).collect() if missing_count > 0 else []
    extra_id_list = extra_ids.rdd.map(lambda x: x.equipment_id).collect() if extra_count > 0 else []

    return {
        "missing_ids": missing_id_list,
        "extra_ids": extra_id_list,
        "all_ids_match": missing_count == 0 and extra_count == 0
    }


def main():
    spark = create_spark_session()

    try:
        # Load data
        sensor_df = spark.read.csv("Datasets/sensor_data.csv", header=True, inferSchema=True)
        equipment_df = spark.read.csv("Datasets/equipment_specs.csv", header=True, inferSchema=True)

        # 1. Verify timestamp range and continuity
        timestamp_results = verify_timestamp_range_and_continuity(sensor_df)
        print("\n1. Timestamp Range and Continuity:")
        print(f"   Min Timestamp: {timestamp_results['min_timestamp']}")
        print(f"   Max Timestamp: {timestamp_results['max_timestamp']}")
        print(f"   Max Gap (seconds): {timestamp_results['max_gap_seconds']}")

        # 2. Check reading ranges
        range_results = check_reading_ranges(sensor_df)
        print("\n2. Reading Ranges:")
        for column, stats in range_results.items():
            print(f"   {column}:")
            print(f"     Min: {stats['min']}")
            print(f"     Max: {stats['max']}")
            print(f"     Negative Count: {stats['negative_count']}")

        # 3. Analyze reading frequency
        frequency_results = analyze_reading_frequency(sensor_df)
        print("\n3. Reading Frequency by Equipment ID:")
        for result in frequency_results[:5]:  # Print first 5 for brevity
            print(f"   Equipment ID: {result['equipment_id']}")
            print(f"     Total Readings: {result['reading_count']}")
            print(f"     Avg Readings/Hour: {result['avg_readings_per_hour']}")
        print("   ...")

        # 4. Verify equipment IDs
        id_verification = verify_equipment_ids(spark, sensor_df, equipment_df)
        print("\n4. Equipment ID Verification:")
        print(f"   All IDs Match: {id_verification['all_ids_match']}")
        print(f"   Missing IDs in Specs: {id_verification['missing_ids']}")
        print(f"   Extra IDs in Specs: {id_verification['extra_ids']}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()