def bar_plot_ordinal(df,intent, intent_column, status_column):
    """
    Plot the distribution of loan status for a given loan intent.

    Parameters:
        intent (str): The loan intent to filter by.
        intent_column (str): The column name for the loan intent.
        status_column (str): The column name for the loan status.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    if intent == "" or intent is None:
        # Show distribution across all intents
        sns.countplot(x=intent_column, hue=df[status_column].astype(str), data=df)
        plt.title(f'Loan Status Distribution by {intent_column}')
        plt.xticks(rotation=45, ha='right')
    else:
        # Show distribution for specific intent
        filtered_df = df[df[intent_column] == intent]
        sns.countplot(x=status_column, data=filtered_df)
        plt.title(f'Loan Status Distribution for {intent_column}={intent}')
    plt.xlabel(status_column if intent else intent_column)
    plt.ylabel('Count')

    # Find the maximum value and its corresponding label
    if intent:
        filtered_df = df[df[intent_column] == intent]
    else:
        filtered_df = df
    value_counts = filtered_df[status_column].value_counts()
    max_value = value_counts.max()
    max_label = value_counts.idxmax()

    # Annotate the maximum value on the bar graph
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        if p.get_height() == max_value:
            plt.gca().annotate(f'{max_label}: {max_value}', (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom', xytext=(0, 20), textcoords='offset points', color='red')

    plt.show()
    
# import pandas as pd
# import numpy as np
# df = pd.read_csv('loan_data.csv')
    
# # # bar_plot_ordinal(df, 'MEDICAL','loan_intent','loan_status')

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

# bar_plot_ordinal(df, '','person_education','loan_status')