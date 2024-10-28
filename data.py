# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset (you can replace with pd.read_csv if you have the dataset file)
# Here we assume the dataset is downloaded locally as "titanic.csv"
df = pd.read_csv('titanic.csv')

# Descriptive statistics
def display_statistics(df):
    """Displays descriptive statistics for the dataset."""
    stats = df.describe()
    skewness = df.skew(numeric_only=True)
    kurtosis = df.kurtosis(numeric_only=True)
    print("Descriptive Statistics:\n", stats)
    print("\nSkewness:\n", skewness)
    print("\nKurtosis:\n", kurtosis)
    return stats, skewness, kurtosis

# Plotting a histogram for age distribution
def plot_age_histogram(df):
    """Generates a histogram for age distribution."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Age', bins=30, kde=True, color="blue")
    plt.title('Age Distribution of Titanic Passengers')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plotting a line plot for average fare by passenger class
def plot_avg_fare_by_class(df):
    """Generates a line plot showing average fare by passenger class."""
    avg_fare_by_class = df.groupby('Pclass')['Fare'].mean()
    avg_fare_by_class.plot(kind='line', marker='o', color='orange', figsize=(8, 6))
    plt.title('Average Fare by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Average Fare')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plotting a box plot for fare distribution across classes
def plot_fare_boxplot(df):
    """Generates a box plot for fare distribution across passenger classes."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Pclass', y='Fare', palette="Set2")
    plt.title('Fare Distribution by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Fare')
    plt.tight_layout()
    plt.show()

# Main function to run all analyses and plots
def main():
    """Main function to execute statistical analysis and generate plots."""
    # Display statistics
    stats, skewness, kurtosis = display_statistics(df)
    
    # Generate plots
    plot_age_histogram(df)
    plot_avg_fare_by_class(df)
    plot_fare_boxplot(df)

# Run the main function
if __name__ == "__main__":
    main()
