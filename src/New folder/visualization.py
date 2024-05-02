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
    condition = cleaned_df[columns_except_std].lt(0.2).any(axis=1)
    filtered_df = cleaned_df[~condition]

    return filtered_df

def plot_data(df):
    """
    Create line plots for the specified numeric columns in the DataFrame.

    Parameters:
    - df: The pandas DataFrame containing the cleaned and filtered data.
    """
    # Set the figure size and layout
    plt.figure(figsize=(15, 10))

    # Plot each of the specified columns
    for column in ['mean', 'median', 'std', 'max', 'p95', 'p90', 'p85']:
        plt.plot(df['date'], df[column], marker='o', label=column)

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
