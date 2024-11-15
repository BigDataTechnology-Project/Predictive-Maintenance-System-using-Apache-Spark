from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, max as spark_max, count, datediff, current_date
import pyspark.sql.functions as F


def create_spark_session():
    return (SparkSession.builder
            .appName("EquipmentSpecsValidation")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.maxResultSize", "4g")
            .getOrCreate())


def validate_equipment_specs(equipment_df):
    # 1. Verify installation_date
    future_installations = equipment_df.filter(col("installation_date") > current_date()).count()
    print(f"1. Number of future installation dates: {future_installations}")

    # 2. Check max values are positive and within reasonable ranges
    max_temp_issues = equipment_df.filter((col("max_temperature") <= 0) | (col("max_temperature") > 1000)).count()
    max_pressure_issues = equipment_df.filter((col("max_pressure") <= 0) | (col("max_pressure") > 10000)).count()
    max_speed_issues = equipment_df.filter(
        (col("max_rotational_speed") <= 0) | (col("max_rotational_speed") > 100000)).count()
    print(f"2. Issues with max values:")
    print(f"   Temperature: {max_temp_issues}, Pressure: {max_pressure_issues}, Rotational Speed: {max_speed_issues}")

    # 3. Check lifetime and warranty periods
    lifetime_issues = equipment_df.filter(
        (col("expected_lifetime_years") <= 0) | (col("expected_lifetime_years") > 100)).count()
    warranty_issues = equipment_df.filter(
        (col("warranty_period_years") <= 0) | (col("warranty_period_years") > 50)).count()
    print(f"3. Issues with periods:")
    print(f"   Lifetime: {lifetime_issues}, Warranty: {warranty_issues}")

    # 4. Verify last_major_overhaul
    future_overhauls = equipment_df.filter(col("last_major_overhaul") > current_date()).count()
    print(f"4. Overhaul issues:")
    print(f"   Future overhauls: {future_overhauls}")
    if future_overhauls > 0:
        print("   Details of future overhauls:")
        current_date_value = datetime.now().date()
        future_overhaul_details = equipment_df.filter(col("last_major_overhaul") > current_date()) \
            .select("equipment_id", "model", "last_major_overhaul", "installation_date",
                    datediff(col("last_major_overhaul"), current_date()).alias("days_in_future"))

        # Convert to Pandas for easier printing
        pandas_df = future_overhaul_details.toPandas()

        # Sort by last_major_overhaul date
        pandas_df = pandas_df.sort_values("last_major_overhaul")

        # Print details
        for index, row in pandas_df.iterrows():
            print(f"     Equipment ID: {row['equipment_id']}")
            print(f"     Model: {row['model']}")
            print(f"     Last Major Overhaul: {row['last_major_overhaul'].date()}")
            print(f"     Installation Date: {row['installation_date'].date()}")
            print(f"     Days in Future: {row['days_in_future']}")
            print("     ---")

        # Print summary statistics
        print("   Summary statistics:")
        print(f"     Earliest future overhaul: {pandas_df['last_major_overhaul'].min().date()}")
        print(f"     Latest future overhaul: {pandas_df['last_major_overhaul'].max().date()}")
        print(f"     Average days in future: {pandas_df['days_in_future'].mean():.2f}")

    invalid_overhauls = equipment_df.filter(col("last_major_overhaul") < col("installation_date")).count()
    print(f"    Before installation: {invalid_overhauls}")


    # 5. Validate criticality categories
    valid_categories = ["High", "Medium", "Low"]
    invalid_criticality = equipment_df.filter(~col("criticality").isin(valid_categories)).count()
    print(f"5. Invalid criticality categories: {invalid_criticality}")

def main():
    spark = create_spark_session()

    try:
        # Load datasets
        equipment_df = spark.read.csv("Datasets/equipment_specs.csv", header=True, inferSchema=True)

        validate_equipment_specs(equipment_df)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()