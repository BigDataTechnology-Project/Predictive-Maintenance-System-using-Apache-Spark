from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, when, count, isnan, isnull, current_date, min, max, datediff
from pyspark.sql.types import StringType


def create_spark_session():
    return (SparkSession.builder
            .appName("MaintenanceLogsValidation")
            .master("local[*]")
            .getOrCreate())


def load_maintenance_logs(spark):
    return spark.read.csv("Datasets/maintenance_logs.csv", header=True, inferSchema=True)

def flag_future_dates(df):
    return df.withColumn(
        "is_future_date",
        when(col("date") > current_date(), True).otherwise(False)
    ).withColumn(
        "days_in_future",
        when(col("date") > current_date(), datediff(col("date"), current_date())).otherwise(0)
    )

def verify_date_range(df):
    date_range = df.select(min("date").alias("min_date"), max("date").alias("max_date")).collect()[0]
    min_date, max_date = date_range["min_date"], date_range["max_date"]
    future_dates = df.filter(col("date") > current_date()).count()
    return min_date, max_date, future_dates


def validate_maintenance_type(df):
    valid_types = ["Routine", "Repair", "Replacement", "Inspection"]
    type_counts = df.groupBy("maintenance_type").count().collect()
    invalid_types = [row["maintenance_type"] for row in type_counts if row["maintenance_type"] not in valid_types]
    return type_counts, invalid_types


def verify_technician_id(df):
    unique_technicians = df.select("technician_id").distinct().count()
    null_technicians = df.filter(col("technician_id").isNull()).count()
    return unique_technicians, null_technicians


def check_positive_values(df):
    negative_duration = df.filter(col("duration_hours") < 0).count()
    negative_cost = df.filter(col("cost") < 0).count()
    return negative_duration, negative_cost


def validate_maintenance_result(df):
    valid_results = ["Successful", "Partial", "Failed"]
    result_counts = df.groupBy("maintenance_result").count().collect()
    invalid_results = [row["maintenance_result"] for row in result_counts if
                       row["maintenance_result"] not in valid_results]
    return result_counts, invalid_results


def check_logical_consistency(df):
    inconsistent_replacements = df.filter(
        (col("maintenance_type") == "Replacement") &
        ((col("parts_replaced").isNull()) | (col("parts_replaced") == ""))
    ).count()
    return inconsistent_replacements


def main():
    spark = create_spark_session()

    try:
        df = load_maintenance_logs(spark)
        df_flagged = flag_future_dates(df)

        # Display counts
        total_count = df_flagged.count()
        future_count = df_flagged.filter(col("is_future_date")).count()
        print(f"Total records: {total_count}")
        print(f"Records with future dates: {future_count}")

        # Display future date records
        print("\nFuture date records:")
        df_flagged.filter(col("is_future_date")).show(truncate=False)

        # 1. Verify date range and check for any future dates
        min_date, max_date, future_dates = verify_date_range(df)
        print(f"Date range: {min_date} to {max_date}")
        print(f"Number of future dates: {future_dates}")

        # 2. Validate maintenance_type categories
        type_counts, invalid_types = validate_maintenance_type(df)
        print("Maintenance type counts:")
        for row in type_counts:
            print(f"  {row['maintenance_type']}: {row['count']}")
        print(f"Invalid maintenance types: {invalid_types}")

        # 3. Verify technician_id exists and is consistent
        unique_technicians, null_technicians = verify_technician_id(df)
        print(f"Number of unique technicians: {unique_technicians}")
        print(f"Number of null technician IDs: {null_technicians}")

        # 4. Ensure duration_hours and cost are positive values
        negative_duration, negative_cost = check_positive_values(df)
        print(f"Number of negative duration_hours: {negative_duration}")
        print(f"Number of negative costs: {negative_cost}")

        # 5. Validate maintenance_result categories
        result_counts, invalid_results = validate_maintenance_result(df)
        print("Maintenance result counts:")
        for row in result_counts:
            print(f"  {row['maintenance_result']}: {row['count']}")
        print(f"Invalid maintenance results: {invalid_results}")

        # 6. Check for logical consistency
        inconsistent_replacements = check_logical_consistency(df)
        print(f"Number of inconsistent replacements: {inconsistent_replacements}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()