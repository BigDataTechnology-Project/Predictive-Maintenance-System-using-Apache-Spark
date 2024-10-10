from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, count, when, isnan, isnull, min, max, datediff, lag, avg, mean, stddev, \
    percentile_approx
from pyspark.sql.types import NumericType, TimestampType, DateType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def load_csv_data(spark, file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)

# Create a Spark session
spark = SparkSession.builder.appName("PredictiveMaintenance").getOrCreate()

# Load the CSV file
csv_file_path = 'C:/Users/KhanhChang/PycharmProjects/Predictive-Maintenance-System-using-Apache-Spark/Datasets/integrated_data.csv'
integrated_df = load_csv_data(spark, csv_file_path)

def initial_data_overview(df):
    # Get the number of rows and columns
    num_rows = df.count()
    num_cols = len(df.columns)
    print(f"Dataset dimensions: {num_rows} rows, {num_cols} columns")

    # Display schema
    print("\nDataset Schema:")
    df.printSchema()

    # Display a few sample records
    print("\nSample Records:")
    df.show(5, truncate=False)

    # Check for missing values
    print("\nMissing Value Count per Column:")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

    # Display summary statistics for numeric columns
    print("\nSummary Statistics for Numeric Columns:")
    df.describe().show()

    # Display unique counts for categorical columns
    categorical_columns = [field.name for field in df.schema.fields if field.dataType.simpleString() == 'string']
    if categorical_columns:
        print("\nUnique Value Counts for Categorical Columns:")
        for col_name in categorical_columns:
            unique_count = df.select(col_name).distinct().count()
            print(f"{col_name}: {unique_count} unique values")


def check_missing_values(df):
    print("Missing Values Check:")

    for c in df.columns:
        column_type = df.schema[c].dataType
        if isinstance(column_type, NumericType):
            # For numeric columns, check for null, NaN, and infinity
            null_count = df.filter(col(c).isNull() | isnan(col(c)) | col(c).isin([float('inf'), float('-inf')])).count()
        elif isinstance(column_type, (TimestampType, DateType)):
            # For date and timestamp columns, only check for null
            null_count = df.filter(col(c).isNull()).count()
        else:
            # For other types (string, boolean, etc.), check for null and empty string
            null_count = df.filter(col(c).isNull() | (col(c) == "")).count()

        print(f"{c}: {null_count} missing values")

def identify_duplicates(df):
    total_records = df.count()
    distinct_records = df.distinct().count()
    duplicates = total_records - distinct_records
    print(f"\nDuplicate Records Check:")
    print(f"Total Records: {total_records}")
    print(f"Distinct Records: {distinct_records}")
    print(f"Duplicate Records: {duplicates}")


def verify_data_types(df):
    print("\nData Types Verification:")
    for field in df.schema.fields:
        print(f"{field.name}: {field.dataType}")


def examine_numerical_ranges(df):
    print("\nNumerical Columns Range:")
    numeric_columns = [field.name for field in df.schema.fields
                       if isinstance(field.dataType, NumericType)]

    if numeric_columns:
        df.select([min(col(c)).alias(f"{c}_min") for c in numeric_columns] +
                  [max(col(c)).alias(f"{c}_max") for c in numeric_columns]) \
            .show(truncate=False)
    else:
        print("No numerical columns found in the dataset.")


def data_quality_assessment(df):
    print("Data Quality Assessment")
    print("=======================")

    check_missing_values(df)
    identify_duplicates(df)
    verify_data_types(df)
    examine_numerical_ranges(df)


def investigate_missing_values(spark, csv_file_path):
    # Load the CSV file
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Columns with missing values
    columns_with_missing = ['production_rate', 'operating_hours', 'downtime_hours', 'operator_id',
                            'product_type', 'raw_material_quality', 'ambient_temperature',
                            'ambient_humidity', 'operation_date']

    # Create a binary column for missing values
    for column in columns_with_missing:
        df = df.withColumn(f"{column}_is_missing", when(col(column).isNull(), 1).otherwise(0))

    print("Consistency of missing values across columns:")
    df.groupBy(*[f"{column}_is_missing" for column in columns_with_missing]) \
        .count() \
        .show()

    print("\nDistribution of missing values across equipment_id:")
    df.groupBy("equipment_id") \
        .agg(*[count(when(col(c).isNull(), c)).alias(f"{c}_missing") for c in columns_with_missing]) \
        .show()

    print("\nDistribution of missing values over time:")
    df.groupBy("timestamp") \
        .agg(*[count(when(col(c).isNull(), c)).alias(f"{c}_missing") for c in columns_with_missing]) \
        .orderBy("timestamp") \
        .show()

def count_rows(df):
    return df.count()

def remove_null_rows(df):
    return df.dropna()


def calculate_numerical_stats(df):
    numeric_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

    stats_exprs = []
    for c in numeric_columns:
        stats_exprs.extend([
            mean(col(c)).alias(f"{c}_mean"),
            percentile_approx(col(c), 0.5).alias(f"{c}_median"),
            stddev(col(c)).alias(f"{c}_stddev"),
            min(col(c)).alias(f"{c}_min"),
            max(col(c)).alias(f"{c}_max")
        ])

    stats = df.select(stats_exprs)
    return stats


def calculate_categorical_freq(df):
    categorical_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

    freq_dists = {}
    for column in categorical_columns:
        freq_dist = df.groupBy(column).agg(count("*").alias("count")) \
            .orderBy("count", ascending=False)
        freq_dists[column] = freq_dist

    return freq_dists


def get_numeric_columns(df):
    return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]


def get_categorical_columns(df):
    return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]


def plot_histograms(df, numeric_columns):
    pdf = df.select(numeric_columns).toPandas()
    num_cols = len(numeric_columns)
    nrows = math.ceil(math.sqrt(num_cols))
    ncols = math.ceil(num_cols / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):
        if i < len(axes):
            pdf[col].hist(ax=axes[i], bins=50)
            axes[i].set_title(col)
            axes[i].set_ylabel('Frequency')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_boxplots(df, numeric_columns):
    pdf = df.select(numeric_columns).toPandas()
    num_cols = len(numeric_columns)
    nrows = math.ceil(math.sqrt(num_cols))
    ncols = math.ceil(num_cols / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):
        if i < len(axes):
            pdf.boxplot(column=col, ax=axes[i])
            axes[i].set_title(col)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def identify_outliers(df, numeric_columns):
    outliers = {}
    for column in numeric_columns:
        stats = df.select(mean(col(column)).alias('mean'),
                          stddev(col(column)).alias('stddev')).collect()[0]
        mean_val, stddev_val = stats['mean'], stats['stddev']
        lower_bound = mean_val - 3 * stddev_val
        upper_bound = mean_val + 3 * stddev_val

        outlier_count = df.filter((col(column) < lower_bound) | (col(column) > upper_bound)).count()
        outliers[column] = outlier_count

    return outliers


def get_numeric_columns(df):
    return [f.name for f in df.schema.fields if f.dataType.typeName() in ["double", "int", "long", "float"]]


# Select only numeric columns
numeric_columns = get_numeric_columns(integrated_df)

# Check for null values
null_counts = integrated_df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in numeric_columns]).collect()[0]
print("Null value counts:")
for col, count in zip(numeric_columns, null_counts):
    print(f"{col}: {count}")

# Remove rows with null values
df_cleaned = integrated_df.dropna(subset=numeric_columns)

# Create vector column of features
vector_col = "correlation_features"
assembler = VectorAssembler(inputCols=numeric_columns, outputCol=vector_col, handleInvalid="skip")
df_vector = assembler.transform(df_cleaned).select(vector_col)

# Compute correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()

# Convert to pandas DataFrame for easier manipulation and visualization
pdf = pd.DataFrame(correlation_matrix, columns=numeric_columns, index=numeric_columns)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(pdf, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Identify strong correlations
strong_correlations = []
for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns)):
        if abs(correlation_matrix[i][j]) > 0.7:  # You can adjust this threshold
            strong_correlations.append((numeric_columns[i], numeric_columns[j], correlation_matrix[i][j]))

# Print strong correlations
print("\nStrong correlations (|correlation| > 0.7):")
for corr in strong_correlations:
    print(f"{corr[0]} - {corr[1]}: {corr[2]:.2f}")

# Optionally, create a heatmap of only strong correlations
if strong_correlations:
    strong_corr_df = pd.DataFrame(strong_correlations, columns=['Feature1', 'Feature2', 'Correlation'])
    strong_corr_matrix = strong_corr_df.pivot(index='Feature1', columns='Feature2', values='Correlation')

    plt.figure(figsize=(10, 8))
    sns.heatmap(strong_corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Strong Correlations Heatmap")
    plt.tight_layout()
    plt.show()



# Call the function
initial_data_overview(integrated_df)
data_quality_assessment(integrated_df)
investigate_missing_values(integrated_df)

remove_null_rows(integrated_df)
print("\nRows with missing values have been removed from the DataFrame.")

# Calculate numerical statistics
numerical_stats = calculate_numerical_stats(integrated_df)

print("\nNumerical Statistics:")
numerical_stats.show(truncate=False)

# Calculate categorical frequency distributions
categorical_freq_dists = calculate_categorical_freq(integrated_df)

print("\nCategorical Frequency Distributions:")
for column, freq_dist in categorical_freq_dists.items():
    print(f"\n{column}:")
    freq_dist.show(10, truncate=False)  # Show top 10 categories

numeric_columns = get_numeric_columns(integrated_df)
categorical_columns = get_categorical_columns(integrated_df)

# Create and display histograms for numerical features
print("Displaying histograms for numerical features...")
plot_histograms(integrated_df, numeric_columns)

# Create and display box plots for numerical features
print("Displaying box plots for numerical features...")
plot_boxplots(integrated_df, numeric_columns)

# Identify outliers
outliers = identify_outliers(integrated_df, numeric_columns)
print("\nOutlier counts (using 3 standard deviations from mean as threshold):")
for column, count in outliers.items():
    print(f"{column}: {count} outliers")

# Remember to stop the Spark session when you're done
spark.stop()