import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jinja2 import Template


DATA_FILE = BASE_DIR / "netflix_titles.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"


def save_bar_chart(series, title, xlabel, ylabel, filename, color="#d81f26"):
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(x=series.values, y=series.index, color=color)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)
    plt.tight_layout()
    path = CHART_DIR / filename
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return f"charts/{filename}"


def save_line_chart(series, title, xlabel, ylabel, filename):
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(x=series.index, y=series.values, marker="o", color="#155e75")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = CHART_DIR / filename
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return f"charts/{filename}"


def save_stacked_chart(table, title, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 6))
    ax = table.plot(kind="bar", stacked=True, figsize=(12, 6), color=["#d81f26", "#155e75"])
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = CHART_DIR / filename
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return f"charts/{filename}"


def clean_dataset(df):
    original_columns = list(df.columns)
    before = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_show_ids": int(df["show_id"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["show_id"]).reset_index(drop=True)

    text_defaults = {
        "director": "Unknown",
        "cast": "Unknown",
        "country": "Unknown",
        "rating": "Unrated",
        "duration": "Unknown",
        "date_added": "Unknown",
    }
    for column, value in text_defaults.items():
        cleaned[column] = cleaned[column].fillna(value).astype(str).str.strip()
        cleaned.loc[cleaned[column] == "", column] = value

    misplaced_duration = cleaned["rating"].str.contains(r"^\d+\s+min$", regex=True, na=False)
    cleaned.loc[misplaced_duration & (cleaned["duration"] == "Unknown"), "duration"] = cleaned.loc[
        misplaced_duration, "rating"
    ]
    cleaned.loc[misplaced_duration, "rating"] = "Unrated"

    cleaned["date_added_clean"] = pd.to_datetime(
        cleaned["date_added"].replace("Unknown", pd.NA).str.strip(),
        errors="coerce",
        format="mixed",
    )
    cleaned["year_added"] = cleaned["date_added_clean"].dt.year
    cleaned["month_added"] = cleaned["date_added_clean"].dt.month_name().fillna("Unknown")
    cleaned["primary_country"] = cleaned["country"].apply(
        lambda value: next((part.strip() for part in str(value).split(",") if part.strip()), "Unknown")
    )
    cleaned["primary_genre"] = cleaned["listed_in"].str.split(",").str[0].str.strip()
    cleaned["duration_number"] = cleaned["duration"].str.extract(r"(\d+)").astype(float)

    after = {
        "rows": len(cleaned),
        "columns": len(cleaned.columns),
        "duplicate_show_ids": int(cleaned["show_id"].duplicated().sum()),
        "missing_values_in_original_columns": int(cleaned[original_columns].isna().sum().sum()),
        "misplaced_durations_fixed": int(misplaced_duration.sum()),
    }
    return cleaned, before, after


def build_report(df, cleaned, before, after, charts, insights, tables):
    template = Template(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Netflix Data Analysis and Visualization Report</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18212f;
      --muted: #607083;
      --line: #d9e1ea;
      --red: #d81f26;
      --blue: #155e75;
      --green: #2f7d4f;
      --bg: #f7f9fb;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.55;
    }
    header {
      background: #111827;
      color: white;
      padding: 34px 48px;
      border-bottom: 6px solid var(--red);
    }
    main { max-width: 1180px; margin: 0 auto; padding: 34px 24px 56px; }
    h1, h2, h3 { margin: 0 0 14px; line-height: 1.2; }
    h1 { font-size: 34px; }
    h2 { font-size: 23px; padding-top: 12px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }
    h3 { font-size: 18px; color: var(--blue); }
    p { margin: 0 0 14px; }
    .subtitle { color: #d3dae6; max-width: 860px; }
    .grid { display: grid; gap: 16px; }
    .metrics { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin: 22px 0; }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    .metric strong { display: block; font-size: 28px; color: var(--red); }
    .metric span { color: var(--muted); font-size: 14px; }
    .two { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 22px; background: white; }
    th, td { padding: 10px 12px; border: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { background: #eef3f7; }
    img { width: 100%; display: block; border: 1px solid var(--line); border-radius: 8px; background: white; }
    .chart { margin: 18px 0 30px; }
    .small { color: var(--muted); font-size: 13px; }
    ul { margin-top: 8px; }
    li { margin-bottom: 8px; }
    @media print {
      body { background: white; }
      header { padding: 24px; }
      main { padding: 24px; }
      .panel, .metric, img { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Netflix Data Analysis and Visualization Report</h1>
    <p class="subtitle">Python mini project using {{ original_file }}. The report follows the assignment steps: load data, preprocess it, perform analytical operations, generate visualizations, and conclude with findings.</p>
  </header>
  <main>
    <section>
      <h2>1. Title</h2>
      <p><strong>Data Analysis and Visualizations on Netflix Titles Dataset</strong></p>
    </section>

    <section>
      <h2>2. Step 1: Load The Data</h2>
      <div class="grid metrics">
        <div class="metric"><strong>{{ row_count }}</strong><span>Total records loaded</span></div>
        <div class="metric"><strong>{{ column_count }}</strong><span>Original columns</span></div>
        <div class="metric"><strong>{{ movie_count }}</strong><span>Movies</span></div>
        <div class="metric"><strong>{{ tv_count }}</strong><span>TV Shows</span></div>
      </div>
      <p>The CSV file contains Netflix catalog records with title type, title name, director, cast, country, date added, release year, rating, duration, genre categories, and description.</p>
      {{ sample_table }}
    </section>

    <section>
      <h2>3. Step 2: Pre-process The Data</h2>
      <p>Cleaning operations performed: removed duplicate IDs, filled missing text fields with meaningful labels, converted date values into date objects, extracted year/month added, extracted primary country and primary genre, and converted duration into a numeric field where possible.</p>
      <div class="grid two">
        <div class="panel">
          <h3>Before Preprocessing</h3>
          {{ before_table }}
        </div>
        <div class="panel">
          <h3>After Preprocessing</h3>
          {{ after_table }}
        </div>
      </div>
      <h3>Missing Values Before Cleaning</h3>
      {{ missing_table }}
    </section>

    <section>
      <h2>4. Step 3: Analytical Operations</h2>
      <p>The selected operations focus on content mix, countries, ratings, genres, release trends, and additions to Netflix over time.</p>
      <div class="grid two">
        <div class="panel">
          <h3>Top Countries</h3>
          {{ tables.top_countries }}
        </div>
        <div class="panel">
          <h3>Top Genres</h3>
          {{ tables.top_genres }}
        </div>
      </div>
      <div class="grid two">
        <div class="panel">
          <h3>Ratings</h3>
          {{ tables.ratings }}
        </div>
        <div class="panel">
          <h3>Recent Release Years</h3>
          {{ tables.release_years }}
        </div>
      </div>
    </section>

    <section>
      <h2>5. Step 4: Visual Analysis</h2>
      {% for chart in charts %}
      <div class="chart">
        <h3>{{ chart.title }}</h3>
        <img src="{{ chart.path }}" alt="{{ chart.title }}">
      </div>
      {% endfor %}
    </section>

    <section>
      <h2>6. Step 5: Front-End Output Screenshots</h2>
      <p>This HTML file is the front-end presentation of the analytical results. The charts above can be used directly as output screenshots for submission.</p>
      <p class="small">Generated files are saved in the output folder: cleaned CSV, charts, summary tables, and this report.</p>
    </section>

    <section>
      <h2>7. Conclusion</h2>
      <ul>
        {% for insight in insights %}
        <li>{{ insight }}</li>
        {% endfor %}
      </ul>
    </section>
  </main>
</body>
</html>
        """
    )

    sample_table = df.head(8).to_html(index=False, classes="table")
    before_table = pd.DataFrame([before]).to_html(index=False)
    after_table = pd.DataFrame([after]).to_html(index=False)
    missing_table = df.isna().sum().rename("missing_count").reset_index().rename(
        columns={"index": "column"}
    ).to_html(index=False)

    html = template.render(
        original_file=DATA_FILE.name,
        row_count=f"{len(df):,}",
        column_count=len(df.columns),
        movie_count=f"{int((cleaned['type'] == 'Movie').sum()):,}",
        tv_count=f"{int((cleaned['type'] == 'TV Show').sum()):,}",
        sample_table=sample_table,
        before_table=before_table,
        after_table=after_table,
        missing_table=missing_table,
        charts=charts,
        insights=insights,
        tables=tables,
    )
    (OUTPUT_DIR / "netflix_analysis_report.html").write_text(html, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATA_FILE)
    cleaned, before, after = clean_dataset(df)
    cleaned.to_csv(OUTPUT_DIR / "netflix_titles_cleaned.csv", index=False)

    type_counts = cleaned["type"].value_counts()
    top_countries = cleaned["primary_country"].value_counts().head(10)
    top_genres = cleaned["listed_in"].str.split(", ").explode().value_counts().head(10)
    top_ratings = cleaned["rating"].value_counts().head(10)
    releases_by_year = cleaned["release_year"].value_counts().sort_index()
    releases_recent = releases_by_year[releases_by_year.index >= 2000]
    added_by_year_type = (
        cleaned.dropna(subset=["year_added"])
        .assign(year_added=lambda x: x["year_added"].astype(int))
        .pivot_table(index="year_added", columns="type", values="show_id", aggfunc="count", fill_value=0)
    )
    added_recent = added_by_year_type[added_by_year_type.index >= 2010]

    charts = [
        {
            "title": "Movies vs TV Shows",
            "path": save_bar_chart(type_counts, "Netflix Content Type Distribution", "Number of titles", "Type", "content_type_distribution.png"),
        },
        {
            "title": "Top 10 Producing Countries",
            "path": save_bar_chart(top_countries, "Top Countries by Number of Netflix Titles", "Number of titles", "Country", "top_countries.png", "#155e75"),
        },
        {
            "title": "Top 10 Genres",
            "path": save_bar_chart(top_genres, "Most Common Netflix Genres", "Number of titles", "Genre", "top_genres.png", "#2f7d4f"),
        },
        {
            "title": "Most Common Ratings",
            "path": save_bar_chart(top_ratings, "Top Content Ratings", "Number of titles", "Rating", "top_ratings.png", "#7c3aed"),
        },
        {
            "title": "Release Trend Since 2000",
            "path": save_line_chart(releases_recent, "Titles by Release Year Since 2000", "Release year", "Number of titles", "release_trend_since_2000.png"),
        },
        {
            "title": "Netflix Additions by Year and Type",
            "path": save_stacked_chart(added_recent, "Netflix Catalog Additions Since 2010", "Year added", "Number of titles", "additions_by_year_type.png"),
        },
    ]

    summary_tables = {
        "top_countries": top_countries.rename("title_count").reset_index().rename(columns={"primary_country": "country"}).to_html(index=False),
        "top_genres": top_genres.rename("title_count").reset_index().rename(columns={"listed_in": "genre"}).to_html(index=False),
        "ratings": top_ratings.rename("title_count").reset_index().rename(columns={"rating": "rating"}).to_html(index=False),
        "release_years": releases_by_year.sort_index(ascending=False).head(10).rename("title_count").reset_index().rename(columns={"release_year": "release_year"}).to_html(index=False),
    }

    insights = [
        f"The dataset contains {len(cleaned):,} records, which is above the required 3,000-record minimum.",
        f"Movies dominate the catalog with {type_counts.get('Movie', 0):,} titles, compared with {type_counts.get('TV Show', 0):,} TV shows.",
        f"The United States is the leading primary country with {top_countries.get('United States', 0):,} titles, followed by India with {top_countries.get('India', 0):,}.",
        f"The most common genre label is {top_genres.index[0]}, appearing in {top_genres.iloc[0]:,} records.",
        f"The most frequent content rating is {top_ratings.index[0]}, with {top_ratings.iloc[0]:,} titles.",
        "Netflix additions increased strongly after 2015, showing rapid catalog expansion in the later years of the dataset.",
        "Preprocessing improved usability by replacing blank categorical values, extracting clean date parts, and creating analysis-friendly country, genre, and duration fields.",
    ]

    pd.DataFrame({"insight": insights}).to_csv(OUTPUT_DIR / "key_insights.csv", index=False)
    build_report(df, cleaned, before, after, charts, insights, summary_tables)

    print("Analysis complete.")
    print(f"Cleaned CSV: {OUTPUT_DIR / 'netflix_titles_cleaned.csv'}")
    print(f"Report: {OUTPUT_DIR / 'netflix_analysis_report.html'}")
    print(f"Charts: {CHART_DIR}")


if __name__ == "__main__":
    main()
