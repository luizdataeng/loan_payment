import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm

"""
Plot a violin plot with annotations for loan interest rate distribution
by loan status, with normality test results and custom legend.

This code generates a violin plot with annotations for the loan interest
rate distribution by loan status, including the mean, median, mode, and
normality test results. The plot also includes a custom legend with mean,
mode, and KDE curve markers.
"""
def plot_violin(loan_status, field_name, df):
    """
    Plot a violin plot with annotations for loan interest rate distribution by loan status, with normality test results and custom legend.

    Parameters:
        loan_status (int): The loan status to filter by (0 = Default, 1 = Paid).
        loan_percent_income (float): The percentage of income allocated for loan repayment.
        df (pandas.DataFrame): The loan data with 'loan_status' and 'loan_int_rate' columns.
    """

    plt.figure(figsize=(12, 7))
    ax = sns.violinplot(
        x="loan_status",
        y=field_name,
        data=df,
        palette={0: "lightcoral", 1: "lightblue"},
        inner="quartile",
        cut=0,
        bw_method=0.2  # Adjust bandwidth for smoother distribution
    )

    # Add distribution curves
    for status in df["loan_status"].unique():
        subset = df[df["loan_status"] == status][field_name].dropna()
        kde = sm.nonparametric.KDEUnivariate(subset)
        kde.fit()
        x_vals = np.linspace(subset.min(), subset.max(), 100)
        y_vals = kde.evaluate(x_vals)
        ax.plot(status + y_vals/y_vals.max()*0.4, x_vals, color='black', lw=1)

    # Annotation parameters
    annotation_config = {
        'Q1': {'color': 'darkblue', 'va': 'bottom', 'ha': 'center'},
        'Med': {'color': 'navy', 'va': 'bottom', 'ha': 'center', 'fontweight': 'bold'},
        'Q3': {'color': 'darkblue', 'va': 'top', 'ha': 'center'},
        'Mean': {'color': 'green', 'va': 'bottom', 'ha': 'left'},
        'Mode': {'color': 'red', 'va': 'top', 'ha': 'right'}
    }

    for i, status in enumerate(df["loan_status"].unique()):
        subset = df[df["loan_status"] == status][field_name].dropna()
        
        # Calculate statistics
        q1, median, q3 = np.percentile(subset, [25, 50, 75])
        mean = subset.mean()
        mode = subset.mode().values[0]
        
        # Add annotations with connecting lines
        for label, value, offset in [('Q1', q1, -0.15), 
                                    ('Med', median, 0), 
                                    ('Q3', q3, 0.15)]:
            ax.plot([i + offset, i], [value, value], color=annotation_config[label]['color'], lw=1)
            ax.text(i + offset, value, f'{label}: {value:.2f}', 
                    **annotation_config[label], fontsize=9)

        # Add mean and mode
        ax.plot(i, mean, 'o', color='green', markersize=8)
        ax.plot(i, mode, 'o', color='red', markersize=8)
        ax.text(i + 0.15, mean, f'Mean: {mean:.2f}', color='green', 
                va='center', fontsize=9)
        ax.text(i - 0.15, mode, f'Mode: {mode:.2f}', color='red', 
                va='center', ha='right', fontsize=9)

    # Add normality test results
    for i, status in enumerate(df["loan_status"].unique()):
        subset = df[df["loan_status"] == status][field_name].dropna()
        _, p_value = stats.normaltest(subset)
        ax.text(i, ax.get_ylim()[1]*0.95, f'Normality p: {p_value:.4f}', 
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

    plt.title("(0 = Não Pagantes | 1 = Pagantes)", pad=20)
    plt.xlabel("Repagamento de Empréstimo", fontsize=12)
    plt.ylabel(field_name, fontsize=12)
    plt.xticks([0, 1], ["Não pagantes", "Pagantes"])
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Mean', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Mode', markerfacecolor='red', markersize=8),
        Line2D([0], [0], color='black', lw=1, label='KDE Curve')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()
    
