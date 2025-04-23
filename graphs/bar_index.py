def bar_plot_index(df, intent_column, status_column):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    """
        Plot a bar chart showing the proportion of payers by education level.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            education_field (str): Field name for the education level.
            status_field (str): Field name for the loan status.
    """

    # Calculate contingency table
    contingency_table = pd.crosstab(
        df['person_education'],
        df['loan_status']
    )

    # Calculate proportion of payers
    prop_payers = contingency_table[1] / contingency_table.sum(axis=1)

    # Plot bar chart
    plt.figure(figsize=(12, 3))
    ax = prop_payers.plot(kind='bar', color='#4caf50', width=0.6, edgecolor='black')

    # Add relative values
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=10
        )

    # Final settings
    plt.title("Pessoas com maior grau de escolaridade tem maior probabilidade de pagar?", pad=10, fontsize=10, fontweight='bold')
    plt.xlabel('\nNível de Educacional', fontsize=10)
    plt.ylabel('Proporção de Pagantes', fontsize=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=8)
    plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# df = pd.read_csv('loan_data.csv')

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

# bar_plot_index(df, 'person_education', 'loan_status')