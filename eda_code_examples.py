"""
Automated Exploratory Data Analysis (EDA) Code Examples

This script demonstrates various approaches to automate EDA using Python libraries:
1. ydata-profiling (formerly pandas-profiling)
2. Sweetviz
3. Custom EDA functions with pandas, matplotlib, and seaborn

Requirements:
- pandas
- numpy
- matplotlib
- seaborn
- ydata-profiling
- sweetviz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz as sv
import os

# Create a directory for saving outputs
os.makedirs('eda_outputs', exist_ok=True)

# Generate a sample dataset for demonstration
def create_sample_dataset(n_samples=1000):
    """Generate a sample dataset with various data types and patterns."""
    np.random.seed(42)
    
    # Create numeric columns with different distributions
    normal_data = np.random.normal(loc=50, scale=15, size=n_samples)
    skewed_data = np.random.exponential(scale=10, size=n_samples)
    uniform_data = np.random.uniform(low=0, high=100, size=n_samples)
    
    # Create categorical data
    categories = ['A', 'B', 'C', 'D']
    categorical_data = np.random.choice(categories, size=n_samples)
    
    # Create binary target variable with some correlation to normal_data
    target_probs = 1 / (1 + np.exp(-(normal_data - 50) / 10))
    target = np.random.binomial(n=1, p=target_probs)
    
    # Create datetime data
    base_date = pd.Timestamp('2023-01-01')
    dates = [base_date + pd.Timedelta(days=i) for i in range(n_samples)]
    
    # Create some missing values
    normal_data_with_missing = normal_data.copy()
    missing_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.1), replace=False)
    normal_data_with_missing[missing_indices] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'normal_feature': normal_data,
        'skewed_feature': skewed_data,
        'uniform_feature': uniform_data,
        'categorical_feature': categorical_data,
        'normal_with_missing': normal_data_with_missing,
        'date': dates,
        'target': target
    })
    
    return df

# Create sample dataset
df = create_sample_dataset()
print("Sample dataset created with shape:", df.shape)
print(df.head())

# Save the dataset for later use
df.to_csv('eda_outputs/sample_dataset.csv', index=False)
print("Sample dataset saved to 'eda_outputs/sample_dataset.csv'")

# Example 1: Automated EDA with ydata-profiling
def run_ydata_profiling(df, output_file='eda_outputs/profile_report.html', minimal=False):
    """Generate a comprehensive EDA report using ydata-profiling."""
    print("\nGenerating ydata-profiling report...")
    
    # Create a profile report
    if minimal:
        profile = ProfileReport(df, minimal=True, title="Minimal Profile Report")
    else:
        profile = ProfileReport(df, title="Comprehensive Profile Report")
    
    # Save the report to an HTML file
    profile.to_file(output_file)
    print(f"ydata-profiling report saved to '{output_file}'")
    
    return profile

# Example 2: Automated EDA with Sweetviz
def run_sweetviz(df, target_col=None, output_file='eda_outputs/sweetviz_report.html'):
    """Generate an EDA report using Sweetviz."""
    print("\nGenerating Sweetviz report...")
    
    # Create a Sweetviz report
    if target_col:
        report = sv.analyze([df, "Dataset"], target_feat=target_col)
    else:
        report = sv.analyze(df)
    
    # Save the report to an HTML file
    report.show_html(output_file)
    print(f"Sweetviz report saved to '{output_file}'")
    
    return report

# Example 3: Custom EDA functions with pandas, matplotlib, and seaborn
class CustomEDA:
    """A class for custom automated EDA using pandas, matplotlib, and seaborn."""
    
    def __init__(self, df, output_dir='eda_outputs'):
        """Initialize with a DataFrame and output directory."""
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Automatically detect data types
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Set plot style
        sns.set(style="whitegrid")
        
    def basic_stats(self, output_file=None):
        """Generate basic statistics for the dataset."""
        print("\nGenerating basic statistics...")
        
        # Summary statistics for numeric columns
        numeric_stats = self.df[self.numeric_cols].describe().T
        numeric_stats['missing'] = self.df[self.numeric_cols].isnull().sum()
        numeric_stats['missing_pct'] = self.df[self.numeric_cols].isnull().mean() * 100
        
        # Summary for categorical columns
        cat_stats = pd.DataFrame({
            'unique_values': [self.df[col].nunique() for col in self.categorical_cols],
            'missing': [self.df[col].isnull().sum() for col in self.categorical_cols],
            'missing_pct': [self.df[col].isnull().mean() * 100 for col in self.categorical_cols]
        }, index=self.categorical_cols)
        
        # Save to file if specified
        if output_file:
            with open(f"{self.output_dir}/{output_file}", 'w') as f:
                f.write("# Numeric Features Statistics\n\n")
                f.write(numeric_stats.to_markdown())
                f.write("\n\n# Categorical Features Statistics\n\n")
                f.write(cat_stats.to_markdown())
        
        return numeric_stats, cat_stats
    
    def plot_distributions(self, max_cols=None):
        """Plot distributions for all numeric features."""
        print("\nPlotting distributions for numeric features...")
        
        # Limit the number of columns if specified
        if max_cols and len(self.numeric_cols) > max_cols:
            plot_cols = self.numeric_cols[:max_cols]
        else:
            plot_cols = self.numeric_cols
        
        # Create a grid of histograms and KDE plots
        n_cols = min(3, len(plot_cols))
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(plot_cols):
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/numeric_distributions.png")
        plt.close()
        
    def plot_categorical(self, max_cols=None):
        """Plot count plots for categorical features."""
        print("\nPlotting distributions for categorical features...")
        
        # Limit the number of columns if specified
        if max_cols and len(self.categorical_cols) > max_cols:
            plot_cols = self.categorical_cols[:max_cols]
        else:
            plot_cols = self.categorical_cols
        
        if not plot_cols:
            print("No categorical columns to plot.")
            return
        
        # Create a grid of count plots
        n_cols = min(2, len(plot_cols))
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(plot_cols):
            # Get value counts and limit to top 10 categories if there are many
            value_counts = self.df[col].value_counts()
            if len(value_counts) > 10:
                top_cats = value_counts.nlargest(10).index
                temp_data = self.df.copy()
                temp_data.loc[~temp_data[col].isin(top_cats), col] = 'Other'
                sns.countplot(y=col, data=temp_data, ax=axes[i], order=temp_data[col].value_counts().index)
                axes[i].set_title(f'Top 10 categories in {col}')
            else:
                sns.countplot(y=col, data=self.df, ax=axes[i], order=self.df[col].value_counts().index)
                axes[i].set_title(f'Categories in {col}')
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/categorical_distributions.png")
        plt.close()
        
    def plot_correlations(self):
        """Plot correlation matrix for numeric features."""
        print("\nPlotting correlation matrix...")
        
        if len(self.numeric_cols) < 2:
            print("Not enough numeric columns for correlation analysis.")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix.png")
        plt.close()
        
    def plot_missing_data(self):
        """Visualize missing data patterns."""
        print("\nPlotting missing data patterns...")
        
        # Calculate missing data percentage
        missing = self.df.isnull().mean() * 100
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            print("No missing data to plot.")
            return
        
        # Plot missing data
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing.index, y=missing.values)
        plt.title('Percentage of Missing Values by Feature')
        plt.xlabel('Features')
        plt.ylabel('Missing Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/missing_data.png")
        plt.close()
        
    def plot_target_relationships(self, target_col):
        """Plot relationships between features and target variable."""
        print(f"\nPlotting relationships with target variable '{target_col}'...")
        
        if target_col not in self.df.columns:
            print(f"Target column '{target_col}' not found in the dataset.")
            return
        
        # For binary target
        if self.df[target_col].nunique() == 2:
            # Numeric features vs binary target
            for col in self.numeric_cols:
                if col != target_col:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=target_col, y=col, data=self.df)
                    plt.title(f'{col} by {target_col}')
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/target_vs_{col}.png")
                    plt.close()
            
            # Categorical features vs binary target
            for col in self.categorical_cols:
                if col != target_col:
                    plt.figure(figsize=(12, 6))
                    
                    # Get top categories if there are many
                    if self.df[col].nunique() > 10:
                        top_cats = self.df[col].value_counts().nlargest(10).index
                        temp_data = self.df.copy()
                        temp_data.loc[~temp_data[col].isin(top_cats), col] = 'Other'
                        
                        # Create a cross-tabulation
                        cross_tab = pd.crosstab(temp_data[col], temp_data[target_col], normalize='index') * 100
                        cross_tab.plot(kind='bar', stacked=True)
                    else:
                        # Create a cross-tabulation
                        cross_tab = pd.crosstab(self.df[col], self.df[target_col], normalize='index') * 100
                        cross_tab.plot(kind='bar', stacked=True)
                    
                    plt.title(f'{target_col} Distribution by {col}')
                    plt.ylabel('Percentage')
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/{col}_vs_target.png")
                    plt.close()
    
    def run_full_eda(self, target_col=None):
        """Run a complete EDA process."""
        print("\nRunning full custom EDA process...")
        
        # Generate basic statistics
        self.basic_stats(output_file="basic_stats.md")
        
        # Plot distributions
        self.plot_distributions()
        self.plot_categorical()
        
        # Plot correlations
        self.plot_correlations()
        
        # Plot missing data
        self.plot_missing_data()
        
        # Plot target relationships if target column is specified
        if target_col:
            self.plot_target_relationships(target_col)
        
        print(f"Custom EDA completed. Results saved to '{self.output_dir}' directory.")

# Run the examples
if __name__ == "__main__":
    print("\n" + "="*50)
    print("AUTOMATED EDA EXAMPLES")
    print("="*50)
    
    # Example 1: ydata-profiling
    profile = run_ydata_profiling(df, output_file='eda_outputs/profile_report.html')
    
    # Example 2: Sweetviz
    sweetviz_report = run_sweetviz(df, target_col='target', output_file='eda_outputs/sweetviz_report.html')
    
    # Example 3: Custom EDA
    custom_eda = CustomEDA(df)
    custom_eda.run_full_eda(target_col='target')
    
    print("\n" + "="*50)
    print("All EDA examples completed successfully!")
    print("="*50)
    print("\nResults are saved in the 'eda_outputs' directory.")
