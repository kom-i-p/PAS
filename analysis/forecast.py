from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "indicators.csv"
VALIDATION_FILE = PROCESSED_DIR / "forecast_validation.csv"
FORECAST_FILE = PROCESSED_DIR / "population_forecast.csv"
METRICS_FILE = PROCESSED_DIR / "forecast_metrics.csv"


TARGET_START_YEAR = 1990
VALIDATION_YEARS = 7
FORECAST_HORIZON = 3


def load_data():
    return pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )


def get_population_years(df):
    years = []

    for column in df.columns:
        try:
            year = int(float(column))
        except ValueError:
            continue

        if 1900 <= year <= 2100:
            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if numeric.notna().any():
                years.append((year, column))

    return sorted(years)


def get_last_known_year(df):
    population_years = get_population_years(df)

    if not population_years:
        raise ValueError(
            "В данных не найдены годовые показатели населения."
        )

    return population_years[-1][0]


def get_population_column(df, year):
    column = str(float(year))

    if column in df.columns:
        return column

    alternative = str(year)

    if alternative in df.columns:
        return alternative

    raise ValueError(
        f"Не найден столбец населения за {year} год."
    )


def prepare_population_long(df):
    population_years = get_population_years(df)

    id_columns = [
        column
        for column in [
            "okato_x",
            "region",
            "federal_district",
            "is_nested",
        ]
        if column in df.columns
    ]

    rows = []

    for _, source_row in df.iterrows():
        for year, column in population_years:
            population = pd.to_numeric(
                source_row[column],
                errors="coerce",
            )

            if pd.isna(population):
                continue

            row = {
                "year": year,
                "population": float(population),
            }

            for id_column in id_columns:
                row[id_column] = source_row[id_column]

            rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "Не удалось сформировать временной ряд населения."
        )

    result = result.sort_values(
        ["region", "year"]
    ).reset_index(drop=True)

    return result


def add_lag_features(data):
    data = data.copy()

    group = data.groupby(
        "region",
        sort=False,
    )["population"]

    for lag in [1, 2, 3]:
        data[f"population_lag_{lag}"] = group.shift(lag)

    return data


def add_rolling_features(data):
    data = data.copy()

    shifted = (
        data.groupby(
            "region",
            sort=False,
        )["population"]
        .shift(1)
    )

    data["population_rolling_mean_3"] = (
        shifted
        .groupby(data["region"])
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    data["population_rolling_std_3"] = (
        shifted
        .groupby(data["region"])
        .rolling(3)
        .std()
        .reset_index(level=0, drop=True)
    )

    return data


def add_demographic_features(source_df, population_data):
    data = population_data.copy()

    demographic_prefixes = [
        "births_",
        "deaths_",
        "natural_change_",
        "birth_rate_",
        "death_rate_",
        "urbanization_",
    ]

    demographic = []

    id_columns = [
        column
        for column in [
            "region",
            "federal_district",
        ]
        if column in source_df.columns
    ]

    for _, row in source_df.iterrows():
        region = row.get("region")

        for year in range(
            TARGET_START_YEAR,
            get_last_known_year(source_df) + 1,
        ):
            record = {
                "region": region,
                "year": year,
            }

            for prefix in demographic_prefixes:
                column = f"{prefix}{year}"

                if column in source_df.columns:
                    record[prefix.rstrip("_")] = pd.to_numeric(
                        row[column],
                        errors="coerce",
                    )

            for column in id_columns:
                record[column] = row[column]

            demographic.append(record)

    demographic = pd.DataFrame(demographic)

    if demographic.empty:
        return data

    data = data.merge(
        demographic,
        on=["region", "year"],
        how="left",
        suffixes=("", "_demographic"),
    )

    demographic_columns = [
        "births",
        "deaths",
        "natural_change",
        "birth_rate",
        "death_rate",
        "urbanization",
    ]

    for column in demographic_columns:
        if column not in data.columns:
            continue

        data[f"{column}_lag_1"] = (
            data.groupby("region")[column].shift(1)
        )

        data[f"{column}_rolling_mean_3"] = (
            data.groupby("region")[column]
            .shift(1)
            .rolling(3)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return data


def build_feature_table(source_df):
    population = prepare_population_long(
        source_df
    )

    population = add_lag_features(
        population
    )

    population = add_rolling_features(
        population
    )

    population = add_demographic_features(
        source_df,
        population,
    )

    population = population.sort_values(
        ["region", "year"]
    ).reset_index(drop=True)

    return population


def get_feature_columns(data):
    excluded = {
        "year",
        "population",
        "okato_x",
    }

    features = []

    for column in data.columns:
        if column in excluded:
            continue

        if column.endswith("_demographic"):
            continue

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            features.append(column)

    categorical = [
        column
        for column in [
            "region",
            "federal_district",
        ]
        if column in data.columns
    ]

    return features + categorical


def split_data(
    data,
    validation_start,
):
    train = data[
        data["year"] < validation_start
    ].copy()

    validation = data[
        data["year"] >= validation_start
    ].copy()

    return train, validation


def build_preprocessor(
    train,
    feature_columns,
):
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            train[column]
        )
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_models(
    train,
    feature_columns,
):
    preprocessor = build_preprocessor(
        train,
        feature_columns,
    )

    models = {
        "Linear Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    non_zero = y_true != 0

    if non_zero.any():
        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    - y_pred[non_zero]
                )
                / y_true[non_zero]
            )
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def evaluate_baseline(validation):
    baseline = validation[
        "population_lag_1"
    ]

    mask = (
        validation["population"].notna()
        & baseline.notna()
    )

    metrics = calculate_metrics(
        validation.loc[mask, "population"].to_numpy(),
        baseline.loc[mask].to_numpy(),
    )

    metrics["model"] = "Naive baseline"

    return metrics


def evaluate_models(
    train,
    validation,
    feature_columns,
):
    X_train = train[feature_columns]
    y_train = train["population"]

    X_validation = validation[feature_columns]
    y_validation = validation["population"]

    models = build_models(
        train,
        feature_columns,
    )

    predictions = {}

    baseline = validation[
        "population_lag_1"
    ].to_numpy()

    predictions["Naive baseline"] = baseline

    results = []

    baseline_metrics = calculate_metrics(
        y_validation.to_numpy(),
        baseline,
    )

    baseline_metrics["model"] = "Naive baseline"

    results.append(baseline_metrics)

    for name, model in models.items():
        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_validation
        )

        predictions[name] = prediction

        metrics = calculate_metrics(
            y_validation.to_numpy(),
            prediction,
        )

        metrics["model"] = name

        results.append(metrics)

    metrics = pd.DataFrame(
        results
    )

    return (
        metrics,
        models,
        predictions,
    )


def save_validation_results(
    validation,
    predictions,
):
    result = validation[
        [
            "region",
            "federal_district",
            "year",
            "population",
        ]
    ].copy()

    for name, prediction in predictions.items():
        model_column = (
            name.lower()
            .replace(" ", "_")
        )

        prediction_column = (
            f"prediction_{model_column}"
        )

        error_column = (
            f"error_{model_column}"
        )

        absolute_error_column = (
            f"absolute_error_{model_column}"
        )

        percentage_error_column = (
            f"percentage_error_{model_column}"
        )

        result[prediction_column] = prediction

        result[error_column] = (
            result["population"]
            - result[prediction_column]
        )

        result[absolute_error_column] = (
            result[error_column].abs()
        )

        result[percentage_error_column] = (
            result[absolute_error_column]
            / result["population"].replace(
                0,
                np.nan,
            )
            * 100
        )

    result.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

def save_metrics(metrics):
    metrics.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

def select_best_model(metrics):
    candidates = metrics[
        metrics["model"]
        != "Naive baseline"
    ]

    best = candidates.sort_values(
        "MAE"
    ).iloc[0]

    return best["model"]


def prepare_future_row(
    history,
    region,
    federal_district,
    year,
):
    region_history = history[
        history["region"] == region
    ].sort_values("year")

    populations = (
        region_history[
            "population"
        ]
        .dropna()
        .tolist()
    )

    if len(populations) < 3:
        return None

    row = {
        "year": year,
        "region": region,
        "federal_district": federal_district,
    }

    row["population_lag_1"] = populations[-1]
    row["population_lag_2"] = populations[-2]
    row["population_lag_3"] = populations[-3]

    last_three = populations[-3:]

    row["population_rolling_mean_3"] = (
        np.mean(last_three)
    )

    row["population_rolling_std_3"] = (
        np.std(last_three, ddof=1)
        if len(last_three) > 1
        else 0
    )

    return row


def forecast_future(
    source_df,
    history,
    models,
    forecast_years,
):
    regions = (
        source_df[
            [
                "region",
                "federal_district",
            ]
        ]
        .drop_duplicates(
            subset=["region"]
        )
    )

    all_forecasts = []

    for model_name, model in models.items():
        model_history = history.copy()

        for year in forecast_years:
            year_rows = []

            for _, region_info in regions.iterrows():
                region = region_info["region"]

                federal_district = (
                    region_info[
                        "federal_district"
                    ]
                )

                row = prepare_future_row(
                    model_history,
                    region,
                    federal_district,
                    year,
                )

                if row is None:
                    continue

                year_rows.append(row)

            if not year_rows:
                continue

            future = pd.DataFrame(
                year_rows
            )

            if model_name == "Naive baseline":
                predictions = future[
                    "population_lag_1"
                ].to_numpy()

            else:
                feature_columns = list(
                    model.named_steps[
                        "preprocessor"
                    ].feature_names_in_
                )

                for column in feature_columns:
                    if column not in future.columns:
                        future[column] = np.nan

                future = future[
                    feature_columns
                ]

                predictions = model.predict(
                    future
                )

            for index, prediction in enumerate(
                predictions
            ):
                region = year_rows[index][
                    "region"
                ]

                federal_district = (
                    year_rows[index][
                        "federal_district"
                    ]
                )

                prediction = max(
                    0,
                    float(prediction),
                )

                all_forecasts.append(
                    {
                        "model": model_name,
                        "region": region,
                        "federal_district":
                            federal_district,
                        "year": year,
                        "population_forecast":
                            prediction,
                    }
                )

                model_history = pd.concat(
                    [
                        model_history,
                        pd.DataFrame(
                            [
                                {
                                    "region":
                                        region,
                                    "year":
                                        year,
                                    "population":
                                        prediction,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    return pd.DataFrame(
        all_forecasts
    )


def add_prediction_intervals(
    forecast,
    validation,
    validation_predictions,
):
    forecast = forecast.copy()

    intervals = {}

    for name, prediction in (
        validation_predictions.items()
    ):
        residuals = (
            validation["population"].to_numpy()
            - prediction
        )

        residual_std = np.std(
            residuals,
            ddof=1,
        )

        intervals[name] = (
            1.96 * residual_std
        )

    forecast["lower_95"] = np.nan
    forecast["upper_95"] = np.nan

    for model_name, interval in intervals.items():
        mask = (
            forecast["model"]
            == model_name
        )

        forecast.loc[
            mask,
            "lower_95"
        ] = (
            forecast.loc[
                mask,
                "population_forecast"
            ]
            - interval
        ).clip(lower=0)

        forecast.loc[
            mask,
            "upper_95"
        ] = (
            forecast.loc[
                mask,
                "population_forecast"
            ]
            + interval
        )

    return forecast


def save_forecast(forecast):
    forecast.to_csv(
        FORECAST_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def reshape_population_forecast():
    forecast = pd.read_csv(
        FORECAST_FILE,
        encoding="utf-8-sig",
    )

    forecast = forecast.drop(
        columns=[
            "lower_95",
            "upper_95",
        ],
        errors="ignore",
    )

    forecast = forecast.pivot(
        index=[
            "region",
            "federal_district",
            "year",
        ],
        columns="model",
        values="population_forecast",
    ).reset_index()

    forecast.columns.name = None

    forecast.to_csv(
        FORECAST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Reshaped forecast saved to {FORECAST_FILE}"
    )


def main():
    df = load_data()

    last_known_year = get_last_known_year(df)
    population_column = get_population_column(
        df,
        last_known_year,
    )

    zero_population_regions = df.loc[
        pd.to_numeric(
            df[population_column],
            errors="coerce",
        ).fillna(0) == 0,
        "region",
    ].tolist()

    df = df.loc[
        ~df["region"].isin(
            zero_population_regions
        )
    ].copy()

    print(
        f"Regions excluded due to zero population "
        f"in {last_known_year}: "
        f"{len(zero_population_regions)}"
    )

    if zero_population_regions:
        print(
            "Excluded regions: "
            + ", ".join(
                sorted(zero_population_regions)
            )
        )

    validation_start = (
        last_known_year
        - VALIDATION_YEARS
        + 1
    )

    training_start = TARGET_START_YEAR

    print(
        f"Last known population year: "
        f"{last_known_year}"
    )

    print(
        f"Training years: "
        f"{training_start}-{validation_start - 1}"
    )

    print(
        f"Validation years: "
        f"{validation_start}-{last_known_year}"
    )

    data = build_feature_table(
        df
    )

    train, validation = split_data(
        data,
        validation_start,
    )

    feature_columns = get_feature_columns(
        train
    )

    required_columns = [
        "population_lag_1",
        "population_lag_2",
        "population_lag_3",
        "population_rolling_mean_3",
        "population_rolling_std_3",
    ]

    feature_columns = [
        column
        for column in feature_columns
        if column in train.columns
    ]

    feature_columns = list(
        dict.fromkeys(
            required_columns
            + feature_columns
        )
    )

    train = train.dropna(
        subset=[
            "population",
            "population_lag_1",
            "population_lag_2",
            "population_lag_3",
        ]
    )

    validation = validation.dropna(
        subset=[
            "population",
            "population_lag_1",
        ]
    )

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    print(
        f"Validation observations: "
        f"{len(validation):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns):,}"
    )

    metrics, models, validation_predictions = (
        evaluate_models(
            train,
            validation,
            feature_columns,
        )
    )

    print("\nValidation metrics:")

    print(
        metrics[
            [
                "model",
                "MAE",
                "RMSE",
                "MAPE",
            ]
        ].to_string(
            index=False
        )
    )

    save_validation_results(
        validation,
        validation_predictions,
    )

    save_metrics(
        metrics
    )

    full_data = data.dropna(
        subset=[
            "population",
            "population_lag_1",
            "population_lag_2",
            "population_lag_3",
        ]
    ).copy()

    for name, model in models.items():
        model.fit(
            full_data[feature_columns],
            full_data["population"],
        )

    forecast_years = list(
        range(
            last_known_year + 1,
            last_known_year
            + FORECAST_HORIZON
            + 1,
        )
    )

    forecast_models = {
        "Naive baseline": None,
        **models,
    }

    forecast = forecast_future(
        df,
        data[
            [
                "region",
                "year",
                "population",
            ]
        ],
        forecast_models,
        forecast_years,
    )

    forecast = add_prediction_intervals(
        forecast,
        validation,
        validation_predictions,
    )

    save_forecast(forecast)
    reshape_population_forecast()

    print(
        f"\nForecast years: "
        f"{forecast_years[0]}-"
        f"{forecast_years[-1]}"
    )

    print(
        f"Forecast observations: "
        f"{len(forecast):,}"
    )

    print(
        f"Validation results -> "
        f"{VALIDATION_FILE}"
    )

    print(
        f"Model metrics -> "
        f"{METRICS_FILE}"
    )

    print(
        f"Forecast -> "
        f"{FORECAST_FILE}"
    )


if __name__ == "__main__":
    main()