# Loan Payment Classification Project
## Setup

To set up the project, create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```



To set up the project, create a virtual environment:

This project analyzes loan payment data using various visualization techniques to understand patterns and relationships in the dataset.

## Visualization Modules

The project includes several specialized visualization modules located in the `graphs` package:

### Bar Plot Modules
- `bar_nominal.py`: Creates bar plots for nominal (categorical) variables
- `bar_unique.py`: Generates bar plots showing unique value distributions
- `bar_ordinal.py`: Visualizes ordinal (ordered categorical) data
- `bar_index.py`: Creates indexed bar plots

### Violin Plot Modules
- `violin.py`: Creates standard violin plots for distribution analysis
- `violin_percent.py`: Specialized violin plots for percentage-based data
- `violin_log.py`: Violin plots with logarithmic scaling
- `visualization_utils.py`: Contains utility functions including `plot_loan_status_violin` for analyzing loan percent income distribution by loan status with statistical annotations (Q1, median, Q3, mean, mode)

## Usage

Import the required visualization modules as needed:

```python
from graphs.bar_nominal import bar_plot_nominal
from graphs.bar_unique import bar_plot_unique_var
from graphs.bar_ordinal import bar_plot_ordinal
from graphs.violin import plot_violin
from graphs.bar_index import bar_plot_index
from graphs.violin_percent import plot_violin_percent
from graphs.violin_log import plot_violin_log
```

Each module provides specialized visualization functions tailored for different types of data analysis in the loan payment classification context.
