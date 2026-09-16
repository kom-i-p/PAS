from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "indicators.csv"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"

VALIDATION_SHARE = 0.2

LAGS = (1, 2, 3)
WINDOW_SIZE = 3

FEATURES = {
    "population": "",
    "births": "births_",
    "deaths": "deaths_",
    "natural_change": "natural_change_",
    "birth_rate": "birth_rate_",
    "death_rate": "death_rate_",
    "urbanization": "urbanization_",
}


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
            columns.append(
                (year, column)
            )

    return sorted(columns)


def get_common_years(df):
    years = []

    for prefix in FEATURES.values():
        current_years = {
            year
            for year, _ in get_year_columns(
                df,
                prefix,
            )
        }

        if not years:
            years = current_years
        else:
            years &= current_years

    return sorted(years)


def transform_to_long_format(df):
    years = get_common_years(df)

    records = []

    for year in years:
        temporary = pd.DataFrame({
            "region": df["region"],
            "federal_district": df["federal_district"],
            "year": year,
        })

        for name, prefix in FEATURES.items():
            columns = dict(
                get_year_columns(
                    df,
                    prefix,
                )
            )

            column = columns.get(year)

            if column is None:
                temporary[name] = pd.NA
            else:
                temporary[name] = df[column]

        records.append(temporary)

    result = pd.concat(
        records,
        ignore_index=True,
    )

    return result.sort_values(
        ["region", "year"]
    ).reset_index(drop=True)


def add_lag_features(df):
    df = df.sort_values(
        ["region", "year"]
    ).copy()

    for feature in FEATURES:
        grouped = df.groupby("region")[feature]

        for lag in LAGS:
            df[
                f"{feature}_lag_{lag}"
            ] = grouped.shift(lag)

    return df


def add_window_features(df):
    df = df.sort_values(
        ["region", "year"]
    ).copy()

    for feature in FEATURES:
        grouped = df.groupby("region")[feature]

        df[
            f"{feature}_rolling_mean_{WINDOW_SIZE}"
        ] = grouped.transform(
            lambda values: (
                values
                .shift(1)
                .rolling(WINDOW_SIZE)
                .mean()
            )
        )

        df[
            f"{feature}_rolling_std_{WINDOW_SIZE}"
        ] = grouped.transform(
            lambda values: (
                values
                .shift(1)
                .rolling(WINDOW_SIZE)
                .std()
            )
        )

    return df


def add_calendar_features(df):
    df = df.copy()

    first_year = df["year"].min()

    df["time_index"] = (
        df["year"] - first_year
    )

    return df


def remove_incomplete_rows(df):
    required_columns = []

    for feature in FEATURES:
        for lag in LAGS:
            required_columns.append(
                f"{feature}_lag_{lag}"
            )

        required_columns.append(
            f"{feature}_rolling_mean_{WINDOW_SIZE}"
        )

    return df.dropna(
        subset=required_columns
    ).copy()


def split_data(df):
    years = sorted(
        df["year"].unique()
    )

    validation_size = max(
        1,
        int(
            len(years)
            * VALIDATION_SHARE
        ),
    )

    split_index = (
        len(years)
        - validation_size
    )

    train_years = years[:split_index]
    validation_years = years[split_index:]

    train = df[
        df["year"].isin(train_years)
    ].copy()

    validation = df[
        df["year"].isin(validation_years)
    ].copy()

    return train, validation


def get_feature_columns(df):
    excluded = {
        "region",
        "federal_district",
        "year",
        "population",
    }

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


def reorder_columns(df):
    feature_columns = get_feature_columns(df)

    return df[
        [
            "region",
            "federal_district",
            "year",
            "population",
            *feature_columns,
        ]
    ]


def save_data(train, validation):
    TRAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Train: {len(train):,} rows "
        f"-> {TRAIN_FILE}"
    )

    print(
        f"Validation: {len(validation):,} rows "
        f"-> {VALIDATION_FILE}"
    )

    print(
        f"Train years: "
        f"{train['year'].min()}-"
        f"{train['year'].max()}"
    )

    print(
        f"Validation years: "
        f"{validation['year'].min()}-"
        f"{validation['year'].max()}"
    )

    print(
        f"Features: "
        f"{len(get_feature_columns(train)):,}"
    )


def main():
    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    df = transform_to_long_format(df)

    print(
        f"Long format: {len(df):,} rows"
    )

    df = add_lag_features(df)
    df = add_window_features(df)
    df = add_calendar_features(df)

    df = remove_incomplete_rows(df)
    df = reorder_columns(df)

    train, validation = split_data(df)

    save_data(
        train,
        validation,
    )


if __name__ == "__main__":
    main()