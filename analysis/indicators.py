from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "analytical_mart.csv"
OUTPUT_FILE = PROCESSED_DIR / "indicators.csv"


def load_data():
    return pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )


def get_year_columns(df, prefix=""):
    columns = []

    for column in df.columns:
        column = str(column)

        if prefix:
            if not column.startswith(prefix):
                continue
            year_part = column[len(prefix):]
        else:
            year_part = column

        try:
            year = int(float(year_part))
        except ValueError:
            continue

        if 1900 <= year <= 2100:
            columns.append((year, column))

    return sorted(columns)


def add_population_indicators(df):
    population_years = get_year_columns(df)

    for index in range(1, len(population_years)):
        year, column = population_years[index]
        previous_year, previous_column = population_years[index - 1]

        df[f"population_change_{year}"] = (
            df[column] - df[previous_column]
        )

        df[f"population_growth_{year}"] = (
            df[f"population_change_{year}"]
            / df[previous_column]
            * 100
        )

    return df


def add_natural_change_indicators(df):
    births_years = get_year_columns(df, "births_")
    deaths_years = get_year_columns(df, "deaths_")

    births = {
        year: column
        for year, column in births_years
    }

    deaths = {
        year: column
        for year, column in deaths_years
    }

    common_years = sorted(
        set(births) & set(deaths)
    )

    for year in common_years:
        df[f"natural_change_{year}"] = (
            df[births[year]] - df[deaths[year]]
        )

    return df


def add_relative_demographic_indicators(df):
    population_years = {
        year: column
        for year, column in get_year_columns(df)
    }

    births_years = {
        year: column
        for year, column in get_year_columns(df, "births_")
    }

    deaths_years = {
        year: column
        for year, column in get_year_columns(df, "deaths_")
    }

    common_years = sorted(
        set(population_years)
        & set(births_years)
        & set(deaths_years)
    )

    for year in common_years:
        population = df[population_years[year]]

        df[f"birth_rate_{year}"] = (
            df[births_years[year]]
            / population
            * 1000
        )

        df[f"death_rate_{year}"] = (
            df[deaths_years[year]]
            / population
            * 1000
        )

    return df


def add_urbanization_indicators(df):
    urban_years = get_year_columns(
        df,
        "urban_share_",
    )

    for year, column in urban_years:
        df[f"urbanization_{year}"] = df[column]

    return df


def save_data(df):
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Indicators: {len(df):,} rows -> {OUTPUT_FILE}"
    )


def main():
    df = load_data()

    df = add_population_indicators(df)
    df = add_natural_change_indicators(df)
    df = add_relative_demographic_indicators(df)
    df = add_urbanization_indicators(df)

    save_data(df)


if __name__ == "__main__":
    main()