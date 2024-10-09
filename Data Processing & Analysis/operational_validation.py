from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, max as spark_max, sum as spark_sum, current_date, datediff, \
    avg as spark_avg, count as spark_count, when, lit, stddev as spark_stddev
from pyspark.sql.window import Window
from pyspark.sql.types import StringType


def create_spark_session():
    return (SparkSession.builder
            .appName("OperationalDataValidation")
            .master("local[*]")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .getOrCreate())


def validate_operational_data(df):
    validation_results = {}

    # 1. Verify date range and check for any future dates
    date_stats = df.agg(
        spark_min("date").alias("min_date"),
        spark_max("date").alias("max_date")
    ).collect()[0]

    validation_results["date_range"] = {
        "min_date": date_stats["min_date"],
        "max_date": date_stats["max_date"]
    }

    future_dates_count = df.filter(col("date") > current_date()).count()
    validation_results["future_dates_count"] = future_dates_count

    # 2. Ensure production_rate is between 0 and 100 (percentage)
    invalid_production_rates = df.filter((col("production_rate") < 0) | (col("production_rate") > 100)).count()
    validation_results["invalid_production_rates_count"] = invalid_production_rates

    # 3. Check if operating_hours and downtime_hours are non-negative and their sum is ≤ 24
    invalid_hours = df.filter(
        (col("operating_hours") < 0) |
        (col("downtime_hours") < 0) |
        (col("operating_hours") + col("downtime_hours") > 24)
    ).count()
    validation_results["invalid_hours_count"] = invalid_hours

    # 4. Validate product_type and raw_material_quality categories
    valid_product_types = ["TypeA", "TypeB", "TypeC"]  # Add all valid types
    valid_quality_levels = ["High", "Medium", "Low"]

    invalid_product_types = df.filter(~col("product_type").isin(valid_product_types)).count()
    invalid_quality_levels = df.filter(~col("raw_material_quality").isin(valid_quality_levels)).count()

    validation_results["invalid_product_types_count"] = invalid_product_types
    validation_results["invalid_quality_levels_count"] = invalid_quality_levels

    # 5. Verify ambient_temperature and ambient_humidity are within reasonable ranges
    # Assuming reasonable ranges: temperature -50°C to 50°C, humidity 0% to 100%
    invalid_temperatures = df.filter((col("ambient_temperature") < -50) | (col("ambient_temperature") > 50)).count()
    invalid_humidity = df.filter((col("ambient_humidity") < 0) | (col("ambient_humidity") > 100)).count()

    validation_results["invalid_temperatures_count"] = invalid_temperatures
    validation_results["invalid_humidity_count"] = invalid_humidity

    # 6. Verify operator_id exists and is consistent
    null_operator_ids = df.filter(col("operator_id").isNull()).count()
    unique_operator_ids = df.select("operator_id").distinct().count()

    validation_results["null_operator_ids_count"] = null_operator_ids
    validation_results["unique_operator_ids_count"] = unique_operator_ids

    return validation_results

def analyze_invalid_hours(df):
    analysis_df = df.withColumn(
        "total_hours", col("operating_hours") + col("downtime_hours")
    ).withColumn(
        "invalid_category",
        when(col("operating_hours") < 0, "negative_operating")
        .when(col("downtime_hours") < 0, "negative_downtime")
        .when(col("total_hours") > 24, "exceeds_24")
        .when(col("total_hours") == 0, "zero_hours")
        .otherwise("valid")
    )

    category_counts = analysis_df.groupBy("invalid_category").count().collect()

    stats = analysis_df.agg(
        spark_min("operating_hours").alias("min_operating"),
        spark_max("operating_hours").alias("max_operating"),
        spark_avg("operating_hours").alias("avg_operating"),
        spark_stddev("operating_hours").alias("stddev_operating"),
        spark_min("downtime_hours").alias("min_downtime"),
        spark_max("downtime_hours").alias("max_downtime"),
        spark_avg("downtime_hours").alias("avg_downtime"),
        spark_stddev("downtime_hours").alias("stddev_downtime")
    ).collect()[0]

    return category_counts, stats


def correct_invalid_hours(df):
    window_spec = Window.partitionBy("equipment_id")

    df_with_avg = df.withColumn(
        "avg_operating_ratio",
        spark_avg(col("operating_hours") / (col("operating_hours") + col("downtime_hours"))).over(window_spec)
    )

    corrected_df = df_with_avg.withColumn(
        "total_hours", col("operating_hours") + col("downtime_hours")
    ).withColumn(
        "operating_hours",
        when(col("total_hours") > 24,
             when(col("avg_operating_ratio").isNotNull(),
                  col("avg_operating_ratio") * 24)
             .otherwise(12))
        .otherwise(col("operating_hours"))
    ).withColumn(
        "downtime_hours",
        when(col("total_hours") > 24,
             when(col("avg_operating_ratio").isNotNull(),
                  (1 - col("avg_operating_ratio")) * 24)
             .otherwise(12))
        .otherwise(col("downtime_hours"))
    )

    corrected_df = corrected_df.drop("avg_operating_ratio", "total_hours")

    return corrected_df

def main():
    spark = create_spark_session()

    try:
        # Load operational data
        operational_df = spark.read.csv("Datasets/operational_data.csv", header=True, inferSchema=True)

        # Perform validation
        validation_results = validate_operational_data(operational_df)

        # Print validation results
        print("\nOperational Data Validation Results:")
        print(
            f"1. Date Range: {validation_results['date_range']['min_date']} to {validation_results['date_range']['max_date']}")
        print(f"   Future Dates Count: {validation_results['future_dates_count']}")
        print(f"2. Invalid Production Rates Count: {validation_results['invalid_production_rates_count']}")
        print(f"3. Invalid Hours Count: {validation_results['invalid_hours_count']}")
        print(f"4. Invalid Product Types Count: {validation_results['invalid_product_types_count']}")
        print(f"   Invalid Quality Levels Count: {validation_results['invalid_quality_levels_count']}")
        print(f"5. Invalid Temperatures Count: {validation_results['invalid_temperatures_count']}")
        print(f"   Invalid Humidity Count: {validation_results['invalid_humidity_count']}")
        print(f"6. Null Operator IDs Count: {validation_results['null_operator_ids_count']}")
        print(f"   Unique Operator IDs Count: {validation_results['unique_operator_ids_count']}")

        # Initial validation and analysis
        initial_validation = validate_operational_data(operational_df)
        print("\nInitial Validation Results:")
        print(f"Invalid Hours Count: {initial_validation['invalid_hours_count']}")

        print("\nDetailed Analysis of Invalid Hours Before Correction:")
        category_counts, stats = analyze_invalid_hours(operational_df)
        for category in category_counts:
            print(f"{category['invalid_category']}: {category['count']}")
        print("\nStatistics:")
        for key, value in stats.asDict().items():
            print(f"{key}: {value}")

        # Correct invalid hours
        corrected_df = correct_invalid_hours(operational_df)

        # Validate corrected data
        corrected_validation = validate_operational_data(corrected_df)
        print("\nValidation Results After Correction:")
        print(f"Invalid Hours Count: {corrected_validation['invalid_hours_count']}")

        # Analyze remaining invalid hours after correction
        if corrected_validation['invalid_hours_count'] > 0:
            print("\nDetailed Analysis of Remaining Invalid Hours After Correction:")
            category_counts, stats = analyze_invalid_hours(corrected_df)
            for category in category_counts:
                print(f"{category['invalid_category']}: {category['count']}")
            print("\nStatistics After Correction:")
            for key, value in stats.asDict().items():
                print(f"{key}: {value}")

        # Save corrected data as a single CSV file
        (corrected_df.coalesce(1).write.csv("Datasets/corrected_operational_data.csv",header=True,mode="overwrite"))
        print("\nCorrected data saved to Datasets/corrected_operational_data.csv")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()