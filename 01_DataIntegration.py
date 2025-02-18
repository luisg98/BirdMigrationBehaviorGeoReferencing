import pandas as pd

def merge_csv_files(file_list, output_file):
    """
    Merges multiple CSV files into a single file.
    
    :param file_list: List of paths to the CSV files to be merged
    :param output_file: Name of the output file
    """
    # List to store DataFrames
    dataframes = []
    
    for file in file_list:
        # Read each CSV file as a DataFrame, treating all data as strings to avoid dtype warnings
        df = pd.read_csv(file, dtype=str, low_memory=False)
        dataframes.append(df)
    
    # Concatenate all DataFrames into one
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the consolidated DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    
    print(f"Files successfully merged! Saved as {output_file}")

# List of CSV files
csv_files = ["Data/Migration of male and female red-backed shrikes from southern Scandinavia (data from Pedersen et al. 2019).csv",
            "Data/Migration of red-backed shrike populations (data from Pedersen et al. 2020).csv",
            "Data/Migration of red-backed shrikes from southern Scandinavia (data from Pedersen et al. 2018).csv",
            "Data/Migration of red-backed shrikes from the Iberian Peninsula (data from Tttrup et al. 2017).csv"
            ]

# Name of the output file
output_csv = "Migration_of_red-backed_shrikes.csv"

# Call the function to merge the files
merge_csv_files(csv_files, output_csv)
