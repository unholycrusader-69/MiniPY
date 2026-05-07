# Netflix Data Analysis and Visualization Mini Project

This project follows the assignment brief from the provided image:

1. Load the CSV dataset.
2. Preprocess and transform the data.
3. Perform analytical operations.
4. Generate visual analysis charts.
5. Present the results in a front-end report.
6. Summarize the conclusion.

## Dataset

- File: `netflix_titles.csv`
- Records: 8,807
- Columns: 12 original columns

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python analyze_netflix.py
```

## Outputs

The script creates these files in `output/`:

- `netflix_titles_cleaned.csv`: cleaned and transformed dataset
- `netflix_analysis_report.html`: front-end report for submission/screenshots
- `key_insights.csv`: conclusion points
- `charts/`: generated visualization images

Open `output/netflix_analysis_report.html` in a browser to view the final report.
