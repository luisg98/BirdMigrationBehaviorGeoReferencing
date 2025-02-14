import pandas as pd

def load_and_filter_data(file_path, output_file):
    """
    Loads the merged dataset and selects specific columns for further analysis.
    
    :param file_path: Path to the merged CSV file
    :param output_file: Name of the filtered output file
    """
    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)
    
    # Select specific columns
    selected_columns = ["timestamp", "location-long", "location-lat", "tag-local-identifier"]
    df = df[selected_columns]
    
    # Count initial rows
    initial_count = len(df)
    
    # Remove rows with missing values in any of the selected columns
    missing_values_count = df.isnull().sum().sum()
    df.dropna(inplace=True)
    after_missing_values_count = len(df)
    
    # Rename columns to more intuitive names
    df.rename(columns={
        "location-lat": "latitude",
        "location-long": "longitude",
        "tag-local-identifier": "bird_id"
    }, inplace=True)
    
    # Remove rows where latitude is exactly 0.001
    outlier_count = (df['latitude'] == 0.001).sum()
    df = df[df['latitude'] != 0.001]
    after_outlier_count = len(df)
    
    # Save the cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved as {output_file}")
    print(f"Initial rows: {initial_count}")
    print(f"Rows removed due to missing values: {initial_count - after_missing_values_count} ({missing_values_count} total missing values)")
    print(f"Rows removed due to latitude outliers (0.001): {outlier_count}")
    print(f"Final rows after cleaning: {after_outlier_count}")

# Input and output file paths
input_csv = "Data/Migration of red-backed shrikes.csv"
output_csv = "Data/Filtered_Migration_Data.csv"

# Call the function to load and filter the dataset
load_and_filter_data(input_csv, output_csv)
