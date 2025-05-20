import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_violin_percent(target, field, df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    plt.violinplot([df[df[target] == status][field].dropna() for status in df[target].unique()],
                   showmeans=True, showmedians=True)
    
    # Add distribution curves
    for i, status in enumerate(df[target].unique(), start=1):
        subset = df[df[target] == status][field].dropna()
        kde = stats.gaussian_kde(subset)
        x_vals = np.linspace(subset.min(), subset.max(), 100)
        y_vals = kde(x_vals)
        ax.plot(i + y_vals/y_vals.max()*0.4, x_vals, color='black', lw=1)

    # Annotation parameters
    annotation_config = {
        'Q1': {'color': 'darkblue', 'va': 'bottom', 'ha': 'center'},
        'Med': {'color': 'navy', 'va': 'bottom', 'ha': 'center', 'fontweight': 'bold'},
        'Q3': {'color': 'darkblue', 'va': 'top', 'ha': 'center'},
        'Mean': {'color': 'green', 'va': 'bottom', 'ha': 'left'},
        'Mode': {'color': 'red', 'va': 'top', 'ha': 'right'}
    }

    for i, status in enumerate(df[target].unique(), start=1):
        subset = df[df[target] == status][field].dropna()
        
        # Calculate statistics
        q1, median, q3 = np.percentile(subset, [25, 50, 75])
        mean = subset.mean()
        mode = subset.mode().values[0]
        
        # Add annotations with connecting lines
        for label, value, offset in [('Q1', q1, -0.15), 
                                    ('Med', median, 0), 
                                    ('Q3', q3, 0.15)]:
            ax.plot([i + offset, i], [value, value], color=annotation_config[label]['color'], lw=1)
            ax.text(i + offset, value, f'{label}: {value:.2f}%', 
                    **annotation_config[label], fontsize=9)

        # Add mean and mode
        ax.plot(i, mean, 'o', color='green', markersize=8)
        ax.plot(i, mode, 'o', color='red', markersize=8)
        ax.text(i + 0.15, mean, f'Mean: {mean:.2f}%', color='green', 
                va='center', fontsize=9)
        ax.text(i - 0.15, mode, f'Mode: {mode:.2f}%', color='red', 
                va='center', ha='right', fontsize=9)

    # Add normality test results
    for i, status in enumerate(df[target].unique(), start=1):
        subset = df[df[target] == status][field].dropna()
        _, p_value = stats.normaltest(subset)
        ax.text(i, ax.get_ylim()[1]*0.95, f'Normality p: {p_value:.4f}', 
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

    plt.title("(0 = Default | 1 = Paid)", pad=20)
    plt.xlabel("Repagamento de Empréstimo", fontsize=12)
    plt.ylabel(field, fontsize=12)
    plt.xticks([1, 2], ["Não pagantes", "Pagantes"])
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

# # Load and prepare data
# csv_path = 'loan_data.csv'
# df = pd.read_csv(csv_path)

# df = df.astype({
#     "person_age": np.dtype("int"),
#     "person_gender": np.dtype("object"),
#     "person_education": np.dtype("object"),
#     "person_income": np.dtype("float"),
#     "person_emp_exp": np.dtype("int"),
#     "person_home_ownership": np.dtype("object"),
#     "loan_amnt": np.dtype("float"),
#     "loan_intent": np.dtype("object"),
#     "loan_int_rate": np.dtype("float"),
#     "loan_percent_income": np.dtype("float"),
#     "cb_person_cred_hist_length": np.dtype("float"),
#     "credit_score": np.dtype("int"),
#     "previous_loan_defaults_on_file": np.dtype("object"),
#     "loan_status": np.dtype("int")
# })

# # Create the violin plot
# plot_violin_percent('loan_status', 'loan_percent_income', df)