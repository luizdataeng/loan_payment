def bar_plot_nominal(df, column,y_column):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    """
        Plot a bar chart with nominal data and a second categorical column.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            column (str): Field name for the column to plot.
            y_column (str): Field name for the second categorical column.
    """

    # Assuming 'df' is your DataFrame and 'loan_status' is a column in it
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, hue=y_column, data=df)
    plt.title(f'{column.capitalize()} Distribution')
    plt.xlabel(column.capitalize())
    plt.ylabel(y_column.capitalize())
    plt.tight_layout()