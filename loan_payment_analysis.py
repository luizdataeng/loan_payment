"""
Loan Payment Analysis

This script performs analysis on loan payment data to understand factors affecting loan status
and build predictive models for loan default prediction.
"""

# 1. Loading Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.api.types import is_integer_dtype, is_object_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
    log_loss,
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    mean_squared_error, 
    r2_score, 
    mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import plotly.express as px
from mlxtend.preprocessing import minmax_scaling
import time

# 2. Data Loading and Preprocessing
def load_data(file_path):
    """Load and prepare the loan payment dataset."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Perform initial data preprocessing."""
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def analyze_features(df):
    """Analyze individual features and their relationships."""
    # Numerical features analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        ax = df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        
        # Add mode annotation
        mode_val = df[col].mode()[0]
        mode_count = df[col].value_counts().max()
        mode_idx = df[col].value_counts().index.get_loc(mode_val)
        
        # Add annotation line and text for mode
        ax.plot([mode_idx-0.2, mode_idx+0.2], [mode_count, mode_count], color='red', lw=2)
        ax.text(mode_idx+0.25, mode_count, f'Mode: {mode_val}\nCount: {mode_count}', 
                ha='left', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()

def main():
    """Main function to run the analysis."""
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Load and preprocess data
    file_path = 'loan_payment_data.csv'  # Update this path as needed
    df = load_data(file_path)
    df = preprocess_data(df)
    
    # Analyze features
    analyze_features(df)

if __name__ == "__main__":
    main()
