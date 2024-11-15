from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, datediff, when, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as F


def create_spark_session():
    return (SparkSession.builder
            .appName("PredictiveMaintenanceDataIntegration")
            .master("local[*]")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .getOrCreate())


def load_data(spark, file_path, format="csv"):
    return spark.read.format(format).option("header", "true").option("inferSchema", "true").load(file_path)


def integrate_data(spark, sensor_data, maintenance_logs, equipment_specs, operational_data):
    # Convert timestamp columns to proper datetime type
    sensor_data = sensor_data.withColumn("timestamp", to_timestamp("timestamp"))
    maintenance_logs = maintenance_logs.withColumn("maintenance_date", to_timestamp("date")).drop("date")
    operational_data = operational_data.withColumn("operation_date", to_timestamp("date")).drop("date")

    # Join sensor data with equipment specs on equipment_id
    integrated_data = sensor_data.join(equipment_specs, "equipment_id", "left")

    # Add latest maintenance information
    window_spec = Window.partitionBy("equipment_id").orderBy(F.desc("maintenance_date"))
    latest_maintenance = maintenance_logs.withColumn("rank", F.row_number().over(window_spec)).filter(
        col("rank") == 1).drop("rank")
    integrated_data = integrated_data.join(latest_maintenance, "equipment_id", "left")

    # Add operational data, joining on equipment_id and date
    integrated_data = integrated_data.join(
        operational_data,
        (integrated_data.equipment_id == operational_data.equipment_id) &
        (F.to_date(integrated_data.timestamp) == F.to_date(operational_data.operation_date)),
        "left"
    ).drop(operational_data.equipment_id)

    # Calculate days since last maintenance
    integrated_data = integrated_data.withColumn(
        "days_since_maintenance",
        when(col("maintenance_date").isNotNull(),
             datediff(col("timestamp"), col("maintenance_date")))
        .otherwise(lit(None))
    )

    # Order by equipment_id
    integrated_data = integrated_data.orderBy("equipment_id")

    # Return all columns
    return integrated_data


def main():
    spark = create_spark_session()

    try:
        # Load datasets
        sensor_data = load_data(spark, "Datasets/sensor_data.csv")
        maintenance_logs = load_data(spark, "Datasets/maintenance_logs.csv")
        equipment_specs = load_data(spark, "Datasets/equipment_specs.csv")
        operational_data = load_data(spark, "Datasets/operational_data_new.csv")

        # Integrate data
        integrated_data = integrate_data(spark, sensor_data, maintenance_logs, equipment_specs, operational_data)

        # Show sample of integrated data
        integrated_data.show(5, truncate=False)

        # Save integrated data as a single CSV file
        (integrated_data.coalesce(1)  # Ensure a single output file
         .write
         .mode("overwrite")
         .option("header", "true")
         .csv("Datasets/integrated_data.csv"))

        print("Data integration completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()