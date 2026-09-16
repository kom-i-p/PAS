from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def load_data():
    path = PROCESSED_DIR / "analytical_mart.csv"

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def print_general_info(df):
    print("=== GENERAL INFORMATION ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nData types:")
    print(df.dtypes.to_string())


def print_missing_values(df):
    print("\n=== MISSING VALUES ===")

    missing = df.isna().sum()
    missing_percent = (
        df.isna().mean() * 100
    )

    result = pd.DataFrame({
        "missing": missing,
        "percent": missing_percent.round(2),
    })

    result = result[result["missing"] > 0]

    if result.empty:
        print("Missing values: none")
        return

    print(result.to_string())


def print_duplicates(df):
    print("\n=== DUPLICATES ===")

    print(
        f"Full duplicates: "
        f"{df.duplicated().sum()}"
    )

    keys = [
        "okato",
        "region",
        "federal_district",
        "is_nested",
    ]

    available_keys = [
        column for column in keys
        if column in df.columns
    ]

    print(
        f"Duplicate regions: "
        f"{df.duplicated(available_keys).sum()}"
    )


def print_numeric_statistics(df):
    print("\n=== NUMERIC STATISTICS ===")

    numeric = df.select_dtypes(
        include="number"
    )

    print(
        numeric.describe()
        .transpose()
        .to_string()
    )


def get_population_columns(df):
    columns = []

    for column in df.columns:
        try:
            year = float(column)
        except (TypeError, ValueError):
            continue

        if year.is_integer() and 1900 <= year <= 2100:
            columns.append(column)

    return sorted(columns, key=lambda column: float(column))


def print_population_statistics(df):
    print("\n=== POPULATION STATISTICS ===")

    population_columns = get_population_columns(df)

    if not population_columns:
        print("Population data not found")
        return

    first_year = population_columns[0]
    last_year = population_columns[-1]

    print(f"First year: {int(float(first_year))}")
    print(f"Last year: {int(float(last_year))}")

    print(f"\nPopulation in {int(float(first_year))}:")
    print(df[first_year].describe().to_string())

    print(f"\nPopulation in {int(float(last_year))}:")
    print(df[last_year].describe().to_string())


def print_population_dynamics(df):
    print("\n=== POPULATION DYNAMICS ===")

    population_columns = get_population_columns(df)

    if len(population_columns) < 2:
        print("Not enough population data")
        return

    first_year = population_columns[0]
    last_year = population_columns[-1]

    dynamics = pd.DataFrame({
        "region": df["region"],
        "start_population": df[first_year],
        "end_population": df[last_year],
    })

    dynamics["change"] = (
        dynamics["end_population"]
        - dynamics["start_population"]
    )

    dynamics["change_percent"] = (
        dynamics["change"]
        / dynamics["start_population"]
        * 100
    )

    print(
        f"\nChange from "
        f"{int(float(first_year))} to "
        f"{int(float(last_year))}:"
    )

    print(
        dynamics[
            [
                "region",
                "start_population",
                "end_population",
                "change_percent",
            ]
        ]
        .sort_values("change_percent", ascending=False)
        .to_string(index=False)
    )


def print_demographic_correlations(df):
    print("\n=== DEMOGRAPHIC CORRELATIONS ===")

    population_years = {
        int(float(column)): column
        for column in get_population_columns(df)
    }

    births_years = {
        int(column.split("_")[1]): column
        for column in df.columns
        if str(column).startswith("births_")
        and str(column).split("_")[1].isdigit()
    }

    deaths_years = {
        int(column.split("_")[1]): column
        for column in df.columns
        if str(column).startswith("deaths_")
        and str(column).split("_")[1].isdigit()
    }

    urban_years = {
        int(column.split("_")[2]): column
        for column in df.columns
        if str(column).startswith("urban_share_")
        and str(column).split("_")[2].isdigit()
    }

    common_years = (
        set(population_years)
        & set(births_years)
        & set(deaths_years)
        & set(urban_years)
    )

    if not common_years:
        print("No common year for all indicators")
        return

    year = max(common_years)

    columns = [
        population_years[year],
        births_years[year],
        deaths_years[year],
        urban_years[year],
    ]

    result = df[columns].copy()

    result.columns = [
        "population",
        "births",
        "deaths",
        "urban_share",
    ]

    print(f"Latest common year: {year}")
    print()
    print(result.corr().round(3).to_string())


def print_osm_coverage(df):
    print("\n=== OSM COVERAGE ===")

    if "osm_name" not in df.columns:
        print("OSM data not found")
        return

    total = len(df)
    matched = df["osm_name"].notna().sum()

    percent = (
        matched / total * 100
        if total
        else 0
    )

    print(f"Fedstat regions: {total}")
    print(f"Matched with OSM: {matched}")
    print(f"Coverage: {percent:.2f}%")

    unmatched = df.loc[
        df["osm_name"].isna(),
        "region",
    ]

    if not unmatched.empty:
        print("\nUnmatched regions:")
        print(
            unmatched
            .drop_duplicates()
            .to_string(index=False)
        )


def print_nested_territories(df):
    print("\n=== NESTED TERRITORIES ===")

    if "is_nested" not in df.columns:
        return

    counts = df["is_nested"].value_counts()

    print(counts.to_string())


def main():
    df = load_data()

    print_general_info(df)
    print_missing_values(df)
    print_duplicates(df)
    print_numeric_statistics(df)
    print_population_statistics(df)
    print_population_dynamics(df)
    print_demographic_correlations(df)
    print_osm_coverage(df)
    print_nested_territories(df)


if __name__ == "__main__":
    main()