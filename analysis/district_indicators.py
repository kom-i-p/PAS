from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "analytical_mart.csv"
OUTPUT_FILE = PROCESSED_DIR / "district_indicators.csv"


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


def aggregate_population(df):
    population_years = get_year_columns(df)

    result = (
        df.groupby("federal_district")[[
            column
            for _, column in population_years
        ]]
        .sum(min_count=1)
        .reset_index()
    )

    return result


def aggregate_births_and_deaths(df, result):
    births_years = dict(
        get_year_columns(df, "births_")
    )

    deaths_years = dict(
        get_year_columns(df, "deaths_")
    )

    for year, column in births_years.items():
        aggregated = (
            df.groupby("federal_district")[column]
            .sum(min_count=1)
        )

        result = result.merge(
            aggregated.rename(f"births_{year}"),
            on="federal_district",
            how="left",
        )

    for year, column in deaths_years.items():
        aggregated = (
            df.groupby("federal_district")[column]
            .sum(min_count=1)
        )

        result = result.merge(
            aggregated.rename(f"deaths_{year}"),
            on="federal_district",
            how="left",
        )

    return result


def aggregate_urbanization(df, result):
    population_years = dict(
        get_year_columns(df)
    )

    urban_years = dict(
        get_year_columns(df, "urban_share_")
    )

    common_years = sorted(
        set(population_years) & set(urban_years)
    )

    grouped = df.groupby("federal_district")

    for year in common_years:
        population = df[population_years[year]]
        urban_share = df[urban_years[year]]

        weighted_urban_population = (
            population * urban_share
        )

        temporary = pd.DataFrame({
            "federal_district": df["federal_district"],
            "population": population,
            "urban_population": weighted_urban_population,
        })

        aggregated = (
            temporary.groupby("federal_district")
            .agg(
                population=("population", "sum"),
                urban_population=("urban_population", "sum"),
            )
        )

        result[f"urbanization_{year}"] = (
            result["federal_district"]
            .map(
                (
                    aggregated["urban_population"]
                    / aggregated["population"]
                )
            )
        )

    return result


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
    births_years = {
        year: column
        for year, column in get_year_columns(df, "births_")
    }

    deaths_years = {
        year: column
        for year, column in get_year_columns(df, "deaths_")
    }

    common_years = sorted(
        set(births_years) & set(deaths_years)
    )

    for year in common_years:
        df[f"natural_change_{year}"] = (
            df[births_years[year]]
            - df[deaths_years[year]]
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


def save_data(df):
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"District indicators: "
        f"{len(df):,} rows -> {OUTPUT_FILE}"
    )


def main():
    df = load_data()

    result = aggregate_population(df)
    result = aggregate_births_and_deaths(df, result)
    result = aggregate_urbanization(df, result)

    result = add_population_indicators(result)
    result = add_natural_change_indicators(result)
    result = add_relative_demographic_indicators(result)

    save_data(result)


if __name__ == "__main__":
    main()