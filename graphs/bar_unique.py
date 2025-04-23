import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
Plot a bar chart showing the proportion of payers by education level,
with annotations showing the number of payers and non-payers.
"""

def bar_plot_unique_var(df, column):
    """
    Plot a bar chart with the maximum value annotated.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Field name for the column to plot.
    """

    # Assuming 'df' is your DataFrame and 'loan_status' is a column in it
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f'{column.capitalize()} Distribution')
    plt.xlabel(column.capitalize())
    plt.ylabel('Count')

    # Find the maximum value and its corresponding label
    max_value = df[column].value_counts().max()
    max_label = df[column].value_counts().idxmax()

    # Annotate the maximum value on the bar graph
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        if p.get_height() == max_value:
            plt.gca().annotate(f'Max: {max_value}', (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='bottom', xytext=(0, 20), textcoords='offset points', color='red')

    plt.show()