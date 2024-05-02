import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean_data(file_path):
    """
    Load data from the given file path, clean it by removing rows with missing values
    in specified columns, and filter rows where certain columns (excluding 'std') 
    have values under 0.2.

    Parameters:
    - file_path: The path to the CSV file containing the data.

    Returns:
    - A cleaned and filtered pandas DataFrame.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Remove rows with missing values in the specified columns
    columns_to_check = ['mean', 'median', 'std', 'max', 'p95', 'p90', 'p85']
    cleaned_df = df.dropna(subset=columns_to_check)

    # Filter rows where columns (excluding 'std') have values under 0.2
    columns_except_std = [col for col in columns_to_check if col != 'std']
    condition = cleaned_df[columns_except_std].lt(0.5).any(axis=1)
    filtered_df = cleaned_df[~condition]

    return filtered_df

def plot_data(df):
    plt.figure(figsize=(15, 10))

    for column in ['mean', 'median', 'std', 'max', 'p95', 'p90', 'p85']:
        plt.plot(df['date'], df[column], marker='o', label=column)

    # Set custom y-axis limits
    # You need to specify appropriate min_value and max_value based on your data's range
    min_value = df[['mean', 'median', 'std', 'max', 'p95', 'p90', 'p85']].min().min() * 0.002  # 80% of the minimum value
    max_value = df[['mean', 'median', 'std', 'max', 'p95', 'p90', 'p85']].max().max() * 2  # 120% of the maximum value

    plt.ylim(min_value, max_value)  # Set the limits

    plt.title("Filtered Data over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def main():
    # Specify the file path to your data file
    file_path = './test_results/curve.csv'

    # Load and clean the data
    df = load_and_clean_data(file_path)

    # Plot the data
    plot_data(df)

if __name__ == "__main__":
    main()
