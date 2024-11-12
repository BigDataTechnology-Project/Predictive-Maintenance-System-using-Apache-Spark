from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull, min, max, mean, stddev, percentile_approx
from pyspark.sql.types import NumericType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os


def create_spark_session():
    """Create and return a Spark session."""
    return (SparkSession.builder
            .appName("EDA")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .getOrCreate())


def load_csv_data(spark, file_path):
    """Load CSV data into a Spark DataFrame."""
    return spark.read.csv(file_path, header=True, inferSchema=True)


def initial_data_overview(df):
    """Provide an initial overview of the dataset."""
    print(f"Dataset dimensions: {df.count()} rows, {len(df.columns)} columns")
    print("\nDataset Schema:")
    df.printSchema()
    print("\nSample Records:")
    df.show(5, truncate=False)
    print("\nSummary Statistics for Numeric Columns:")
    df.describe().show()

    categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
    if categorical_columns:
        print("\nUnique Value Counts for Categorical Columns:")
        for col_name in categorical_columns:
            unique_count = df.select(col_name).distinct().count()
            print(f"{col_name}: {unique_count} unique values")


def check_missing_values(df):
    """Check for missing values in each column, remove them, and check again."""
    print("Missing Values Check (Before Removal):")
    for c in df.columns:
        column_type = df.schema[c].dataType
        if isinstance(column_type, NumericType):
            null_count = df.filter(col(c).isNull() | isnan(col(c)) | col(c).isin([float('inf'), float('-inf')])).count()
        else:
            null_count = df.filter(col(c).isNull() | (col(c) == "")).count()
        print(f"{c}: {null_count} missing values")

    # Remove rows with missing values
    df_cleaned = df.na.drop()
    print("\nRows with missing values have been removed.")

    print("\nMissing Values Check (After Removal):")
    for c in df_cleaned.columns:
        column_type = df_cleaned.schema[c].dataType
        if isinstance(column_type, NumericType):
            null_count = df_cleaned.filter(
                col(c).isNull() | isnan(col(c)) | col(c).isin([float('inf'), float('-inf')])).count()
        else:
            null_count = df_cleaned.filter(col(c).isNull() | (col(c) == "")).count()
        print(f"{c}: {null_count} missing values")

    return df_cleaned


def identify_duplicates(df):
    """Identify duplicate records in the dataset."""
    total_records = df.count()
    distinct_records = df.distinct().count()
    duplicates = total_records - distinct_records
    print(f"\nDuplicate Records Check:")
    print(f"Total Records: {total_records}")
    print(f"Distinct Records: {distinct_records}")
    print(f"Duplicate Records: {duplicates}")


def verify_data_types(df):
    """Verify data types of each column."""
    print("\nData Types Verification:")
    for field in df.schema.fields:
        print(f"{field.name}: {field.dataType}")


def examine_numerical_ranges(df):
    """Examine the range of values in numerical columns."""
    print("\nNumerical Columns Range:")
    numeric_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]
    if numeric_columns:
        df.select([min(col(c)).alias(f"{c}_min") for c in numeric_columns] +
                  [max(col(c)).alias(f"{c}_max") for c in numeric_columns]).show(truncate=False)
    else:
        print("No numerical columns found in the dataset.")


def calculate_categorical_freq(df):
    """Calculate and print frequency distributions for categorical columns."""
    categorical_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    print("\nCategorical Frequency Distributions:")
    for column in categorical_columns:
        freq_dist = df.groupBy(column).agg(count("*").alias("count")).orderBy("count", ascending=False)
        print(f"\n{column}:")
        freq_dist.show(10, truncate=False)

def identify_outliers(df, numeric_columns):
    """Identify and print outliers in numerical columns."""
    print("\nOutlier counts (using 3 standard deviations from mean as threshold):")
    for column in numeric_columns:
        stats = df.select(mean(col(column)).alias('mean'),
                          stddev(col(column)).alias('stddev')).collect()[0]
        mean_val, stddev_val = stats['mean'], stats['stddev']
        lower_bound = mean_val - 3 * stddev_val
        upper_bound = mean_val + 3 * stddev_val
        outlier_count = df.filter((col(column) < lower_bound) | (col(column) > upper_bound)).count()
        print(f"{column}: {outlier_count} outliers")


def compute_correlations(df, numeric_columns):
    """Compute correlations and print strong and weak correlations."""
    vector_col = "correlation_features"
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol=vector_col, handleInvalid="skip")
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
    correlation_matrix = matrix.toArray().tolist()

    strong_correlations = []
    weak_correlations = []
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            correlation = correlation_matrix[i][j]
            if abs(correlation) > 0.7:
                strong_correlations.append((numeric_columns[i], numeric_columns[j], correlation))
            elif abs(correlation) < 0.3:
                weak_correlations.append((numeric_columns[i], numeric_columns[j], correlation))

    print("\nStrong correlations (|correlation| > 0.7):")
    for corr in strong_correlations:
        print(f"{corr[0]} - {corr[1]}: {corr[2]:.2f}")

    print("\nWeak correlations (|correlation| < 0.3):")
    for corr in weak_correlations:
        print(f"{corr[0]} - {corr[1]}: {corr[2]:.2f}")


def save_cleaned_dataset(df, output_dir, format='csv'):
    """Save the cleaned dataset in CSV format."""
    output_path = os.path.join(output_dir, f'cleaned_dataset.{format}')

    if format == 'csv':
        # Save as a single CSV file
        df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

        # Rename the output file
        csv_file = [f for f in os.listdir(output_path) if f.endswith('.csv')][0]
        os.rename(os.path.join(output_path, csv_file),
                  os.path.join(output_path, 'cleaned_dataset.csv'))

        print(f"Cleaned dataset saved to {os.path.join(output_path, 'cleaned_dataset.csv')}")
    else:
        df.write.format(format).mode('overwrite').save(output_path)
        print(f"Cleaned dataset saved to {output_path}")


def main():
    spark = create_spark_session()

    try:
        # Load data
        csv_file_path = 'Datasets/integrated_data.csv'
        integrated_df = load_csv_data(spark, csv_file_path)

        # Print số lượng bản ghi ban đầu
        total_records_before = integrated_df.count()
        print(f"\nTotal records before filtering: {total_records_before}")

        # Remove invalid records (where timestamp < installation_date)
        df_valid = integrated_df.filter(F.col("timestamp") >= F.col("installation_date"))

        # Print số lượng bản ghi sau khi lọc
        total_records_after = df_valid.count()
        print(f"Total records after filtering: {total_records_after}")
        print(f"Removed {total_records_before - total_records_after} records")

        # Perform initial data overview
        initial_data_overview(df_valid)

        # Check missing values and remove them
        df_cleaned = check_missing_values(df_valid)

        # Identify duplicates
        identify_duplicates(df_cleaned)

        # Verify data types
        verify_data_types(df_cleaned)

        # Examine numerical ranges
        examine_numerical_ranges(df_cleaned)

        # Calculate categorical frequency distributions
        calculate_categorical_freq(df_cleaned)

        # Get numeric columns
        numeric_columns = [f.name for f in df_cleaned.schema.fields if isinstance(f.dataType, NumericType)]

        # Set output directory
        output_dir = 'Data Processing & Analysis/DEA & Feature Engineering'

        # Identify outliers
        identify_outliers(df_cleaned, numeric_columns)

        # Compute correlations
        compute_correlations(df_cleaned, numeric_columns)

        # Save the cleaned dataset as CSV
        save_cleaned_dataset(df_cleaned, output_dir, format='csv')

    finally:
        spark.stop()


if __name__ == "__main__":
    main()