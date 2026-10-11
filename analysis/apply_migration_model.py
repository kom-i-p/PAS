import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "data" / "models" / "migration"
PAIR_MATRICES_DIR = MODEL_DIR / "subject_matrices_by_district_pair"

COMBINED_MATRICES_DIR = MODEL_DIR / "combined_matrices"

TRAINING_DATA_PATH = PROCESSED_DIR / "migration_training.csv"
SUBJECT_INDICATORS_PATH = PROCESSED_DIR / "indicators.csv"
SUBJECT_REACHABILITY_PATH = (
    PROCESSED_DIR / "subject_reachability_matrix.csv"
)
DISTRICT_MIGRATION_PATH = PROCESSED_DIR / "migration.csv"

MODEL_PATH = MODEL_DIR / "random_forest_full.joblib"
PREDICTIONS_PATH = MODEL_DIR / "subject_migration_predictions.csv"
MANIFEST_PATH = MODEL_DIR / "district_pair_matrices_manifest.csv"
METADATA_PATH = MODEL_DIR / "application_metadata.json"

RANDOM_STATE = 42

DISTRICTS = [
    "Дальневосточный",
    "Приволжский",
    "Северо-Западный",
    "Северо-Кавказский",
    "Сибирский",
    "Уральский",
    "Центральный",
    "Южный",
]

DEMOGRAPHIC_FEATURES = [
    "population",
    "births",
    "deaths",
    "urbanization",
    "population_change",
    "population_growth",
    "natural_change",
    "birth_rate",
    "death_rate",
]

FEATURE_COLUMNS = (
    ["reachability"]
    + [
        f"{prefix}_{feature}"
        for prefix in ("source", "destination")
        for feature in DEMOGRAPHIC_FEATURES
    ]
)


def normalize_name(value):
    value = str(value).strip().lower()
    value = value.replace("ё", "е")
    value = re.sub(r"^\d+\s+", "", value)
    return re.sub(r"[^а-яa-z0-9]", "", value)


def normalize_district(value):
    value = normalize_name(value)
    value = value.replace("федеральныйокруг", "")
    value = value.replace("округ", "")

    aliases = {
        normalize_name(district): district
        for district in DISTRICTS
    }

    if value in aliases:
        return aliases[value]

    raise ValueError(
        f"Не удалось сопоставить федеральный округ: {value}"
    )


def find_subject_column(indicators, subject_names):
    target_names = {
        normalize_name(name)
        for name in subject_names
    }

    best_column = None
    best_matches = set()

    for column in indicators.columns:
        values = indicators[column].dropna().astype(str)
        normalized_values = {
            normalize_name(value)
            for value in values
        }

        matches = target_names.intersection(normalized_values)

        if len(matches) > len(best_matches):
            best_column = column
            best_matches = matches

    if best_column is None or best_matches != target_names:
        missing = sorted(target_names - best_matches)

        raise ValueError(
            "Не удалось полностью сопоставить субъекты из матрицы "
            "доступности с indicators.csv. "
            f"Найдено: {len(best_matches)} из {len(target_names)}. "
            f"Примеры ненайденных названий: {missing[:10]}"
        )

    return best_column


def find_district_column(indicators, subject_column, subject_names):
    subject_keys = {
        normalize_name(name): name
        for name in subject_names
    }

    if len(subject_keys) != len(subject_names):
        raise ValueError(
            "Названия субъектов из матрицы не уникальны "
            "после нормализации."
        )

    district_aliases = {
        normalize_name(district): district
        for district in DISTRICTS
    }

    subject_keys_set = set(subject_keys)
    candidates = []

    for column in indicators.columns:
        if column == subject_column:
            continue

        assignments = {}
        conflicts = {}

        for _, row in indicators[
            [subject_column, column]
        ].iterrows():
            raw_subject = row[subject_column]

            if pd.isna(raw_subject):
                continue

            subject_key = normalize_name(str(raw_subject))

            if subject_key not in subject_keys_set:
                continue

            raw_district = row[column]

            if pd.isna(raw_district):
                continue

            district_key = normalize_name(str(raw_district))
            district_key = (
                district_key
                .replace("федеральныйокруг", "")
                .replace("округ", "")
            )

            if district_key not in district_aliases:
                continue

            district = district_aliases[district_key]

            if subject_key not in assignments:
                assignments[subject_key] = set()

            assignments[subject_key].add(district)

            if len(assignments[subject_key]) > 1:
                conflicts[subject_key] = assignments[
                    subject_key
                ]

        missing_subjects = (
            subject_keys_set - set(assignments)
        )

        if not missing_subjects and not conflicts:
            candidates.append(column)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise ValueError(
            "Найдено несколько столбцов с федеральными округами: "
            f"{candidates}. Необходимо указать правильный столбец."
        )

    diagnostics = []

    for column in indicators.columns:
        if column == subject_column:
            continue

        matched_subjects = set()
        district_assignments = {}

        for _, row in indicators[
            [subject_column, column]
        ].iterrows():
            raw_subject = row[subject_column]

            if pd.isna(raw_subject):
                continue

            subject_key = normalize_name(str(raw_subject))

            if subject_key not in subject_keys_set:
                continue

            raw_district = row[column]

            if pd.isna(raw_district):
                continue

            district_key = normalize_name(str(raw_district))
            district_key = (
                district_key
                .replace("федеральныйокруг", "")
                .replace("округ", "")
            )

            if district_key not in district_aliases:
                continue

            matched_subjects.add(subject_key)

            district = district_aliases[district_key]
            district_assignments.setdefault(
                subject_key, set()
            ).add(district)

        if matched_subjects:
            conflicts = {
                subject_keys[key]: sorted(values)
                for key, values in district_assignments.items()
                if len(values) > 1
            }

            diagnostics.append({
                "column": column,
                "matched_subjects": len(matched_subjects),
                "missing_subjects": [
                    subject_keys[key]
                    for key in sorted(
                        subject_keys_set - matched_subjects
                    )
                ][:10],
                "conflicts": conflicts,
            })

    raise ValueError(
        "Не удалось однозначно определить столбец "
        "с федеральными округами для субъектов из матрицы.\n"
        f"Ожидается субъектов: {len(subject_names)}.\n"
        f"Подходящие столбцы и диагностика: {diagnostics}"
    )


def find_population_column(columns, year):
    for column in columns:
        try:
            if float(str(column).strip()) == float(year):
                return column
        except (TypeError, ValueError):
            continue

    return None


def normalize_column_name(value):
    value = str(value).strip().lower()
    value = value.replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]", "", value)


def find_indicator_column(columns, feature, year):
    if feature == "population":
        return find_population_column(columns, year)

    expected_names = [
        f"{feature}_{year}",
        f"{feature}{year}",
        f"{year}_{feature}",
        f"{year}{feature}",
    ]

    normalized_columns = {
        column: normalize_column_name(column)
        for column in columns
    }

    for expected in expected_names:
        expected_normalized = normalize_column_name(expected)

        for column, normalized in normalized_columns.items():
            if normalized == expected_normalized:
                return column

    feature_normalized = normalize_column_name(feature)

    matches = [
        column
        for column, normalized in normalized_columns.items()
        if feature_normalized in normalized
        and str(year) in normalized
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def build_indicator_column_map(columns, year):
    result = {}

    for feature in DEMOGRAPHIC_FEATURES:
        column = find_indicator_column(
            columns=columns,
            feature=feature,
            year=year,
        )

        if column is None:
            return None

        result[feature] = column

    return result


def diagnose_missing_indicators(columns, years):
    print("\nОтсутствующие демографические индикаторы:")

    for year in sorted(years):
        missing_features = []

        for feature in DEMOGRAPHIC_FEATURES:
            column = find_indicator_column(
                columns=columns,
                feature=feature,
                year=year,
            )

            if column is None:
                missing_features.append(feature)

        if missing_features:
            print(f"{year}: {', '.join(missing_features)}")



def select_feature_year(
    indicators,
    subject_column,
    subject_names,
    max_year,
):
    subject_keys = {
        normalize_name(name)
        for name in subject_names
    }

    subject_indicators = indicators.loc[
        indicators[subject_column].map(normalize_name).isin(
            subject_keys
        )
    ].copy()

    subject_indicators["_subject_key"] = (
        subject_indicators[subject_column].map(normalize_name)
    )

    if subject_indicators["_subject_key"].duplicated().any():
        duplicates = subject_indicators.loc[
            subject_indicators["_subject_key"].duplicated(
                keep=False
            ),
            subject_column,
        ].astype(str).tolist()

        raise ValueError(
            "Найдены повторяющиеся записи субъектов: "
            f"{duplicates[:10]}"
        )

    column_map = build_indicator_column_map(
        columns=indicators.columns,
        year=max_year,
    )

    if column_map is None:
        raise ValueError(
            f"Для {max_year} года отсутствуют столбцы "
            "одного или нескольких демографических признаков."
        )

    indexed = subject_indicators.set_index("_subject_key")

    values = pd.DataFrame(
        {
            feature: pd.to_numeric(
                indexed[column],
                errors="coerce",
            )
            for feature, column in column_map.items()
        },
        index=indexed.index,
    )

    valid_mask = values.notna().all(axis=1)
    valid_keys = set(values.index[valid_mask])

    filtered_indicators = subject_indicators.loc[
        subject_indicators["_subject_key"].isin(valid_keys)
    ].copy()

    if filtered_indicators.empty:
        raise ValueError(
            f"В {max_year} году нет субъектов, "
            "у которых заполнены все демографические признаки."
        )

    return max_year, column_map, filtered_indicators


def build_subject_data(
    subject_names,
    subject_indicators,
    subject_column,
    district_column,
    column_map,
    reachability,
    indicators,
):
    indexed = subject_indicators.copy()

    indexed["_subject_key"] = (
        indexed[subject_column]
        .astype(str)
        .map(normalize_name)
    )

    if indexed["_subject_key"].duplicated().any():
        raise ValueError(
            "После фильтрации обнаружены повторные записи субъектов."
        )

    indexed = indexed.set_index("_subject_key")

    expected_subjects = {
        normalize_name(name): name
        for name in subject_names
    }

    found_subjects = set(indexed.index)

    missing_subjects = (
        set(expected_subjects) - found_subjects
    )

    if missing_subjects:
        raise ValueError(
            "В годовом наборе отсутствуют ожидаемые субъекты:\n"
            + "\n".join(
                expected_subjects[key]
                for key in sorted(missing_subjects)
            )
        )

    features_by_subject = {}
    district_by_subject = {}

    for subject in subject_names:
        key = normalize_name(subject)
        row = indexed.loc[key]

        features = {}

        for feature, column in column_map.items():
            value = pd.to_numeric(
                pd.Series([row[column]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(value):
                raise ValueError(
                    f"У субъекта {subject} отсутствует "
                    f"значение признака {feature}."
                )

            features[feature] = float(value)

        features_by_subject[subject] = features

        district_by_subject[subject] = normalize_district(
            row[district_column]
        )

    empty_districts = [
        district
        for district in DISTRICTS
        if not any(
            district_by_subject[subject] == district
            for subject in subject_names
        )
    ]

    if empty_districts:
        raise ValueError(
            "После фильтрации в следующих федеральных округах "
            "не осталось субъектов: "
            + ", ".join(empty_districts)
        )

    return features_by_subject, district_by_subject


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)

    total = scores.sum()

    if total <= 0:
        return np.full(
            len(scores),
            1.0 / len(scores),
            dtype=float,
        )

    return scores / total


def load_district_flows(path, feature_year):
    migration = pd.read_csv(path)

    required_columns = {
        "year",
        "source_federal_district",
        "destination_federal_district",
        "migration",
    }

    missing = required_columns - set(migration.columns)

    if missing:
        raise ValueError(
            "В migration.csv отсутствуют необходимые столбцы: "
            f"{sorted(missing)}"
        )

    migration["year"] = pd.to_numeric(
        migration["year"],
        errors="coerce",
    )

    migration["migration"] = pd.to_numeric(
        migration["migration"],
        errors="coerce",
    )

    available_years = sorted(
        migration.loc[
            migration["year"].notna()
            & (migration["year"] <= feature_year),
            "year",
        ].astype(int).unique().tolist()
    )

    if not available_years:
        raise ValueError(
            "В migration.csv нет годов, не превышающих год "
            "демографических признаков."
        )

    migration_year = (
        feature_year
        if feature_year in available_years
        else max(available_years)
    )

    year_data = migration.loc[
        migration["year"] == migration_year
    ].copy()

    year_data["source_district"] = (
        year_data["source_federal_district"].map(normalize_district)
    )

    year_data["destination_district"] = (
        year_data["destination_federal_district"].map(
            normalize_district
        )
    )

    if year_data.duplicated(
        ["source_district", "destination_district"]
    ).any():
        raise ValueError(
            f"В migration.csv есть повторяющиеся пары округов "
            f"за {migration_year} год."
        )

    if year_data["migration"].isna().any():
        raise ValueError(
            f"В migration.csv есть пропущенные значения миграции "
            f"за {migration_year} год."
        )

    flows = {}

    for row in year_data.itertuples(index=False):
        source = row.source_district
        destination = row.destination_district
        value = float(row.migration)

        if value < 0:
            raise ValueError(
                "Обнаружен отрицательный объём миграции: "
                f"{source} → {destination}, {value}"
            )

        flows[(source, destination)] = value

    expected_pairs = {
        (source, destination)
        for source in DISTRICTS
        for destination in DISTRICTS
    }

    missing_pairs = expected_pairs - set(flows)

    if missing_pairs:
        raise ValueError(
            f"В migration.csv за {migration_year} год отсутствуют "
            f"{len(missing_pairs)} пар округов. "
            f"Примеры: {sorted(missing_pairs)[:10]}"
        )

    return migration_year, flows


def load_district_flows_for_year(path, migration_year):
    migration = pd.read_csv(path)

    required_columns = {
        "year",
        "source_federal_district",
        "destination_federal_district",
        "migration",
    }

    missing = required_columns - set(migration.columns)

    if missing:
        raise ValueError(
            "В migration.csv отсутствуют необходимые столбцы: "
            f"{sorted(missing)}"
        )

    migration["year"] = pd.to_numeric(
        migration["year"],
        errors="coerce",
    )

    migration["migration"] = pd.to_numeric(
        migration["migration"],
        errors="coerce",
    )

    year_data = migration.loc[
        migration["year"] == migration_year
    ].copy()

    if year_data.empty:
        raise ValueError(
            f"В migration.csv нет данных за {migration_year} год."
        )

    year_data["source_district"] = (
        year_data["source_federal_district"].map(normalize_district)
    )

    year_data["destination_district"] = (
        year_data["destination_federal_district"].map(
            normalize_district
        )
    )

    if year_data.duplicated(
        ["source_district", "destination_district"]
    ).any():
        raise ValueError(
            f"В migration.csv есть повторяющиеся пары округов "
            f"за {migration_year} год."
        )

    if year_data["migration"].isna().any():
        raise ValueError(
            f"В migration.csv есть пропущенные значения миграции "
            f"за {migration_year} год."
        )

    flows = {}

    for row in year_data.itertuples(index=False):
        source = row.source_district
        destination = row.destination_district
        value = float(row.migration)

        if value < 0:
            raise ValueError(
                "Обнаружен отрицательный объём миграции: "
                f"{source} → {destination}, {value}"
            )

        flows[(source, destination)] = value

    expected_pairs = {
        (source, destination)
        for source in DISTRICTS
        for destination in DISTRICTS
    }

    missing_pairs = expected_pairs - set(flows)

    if missing_pairs:
        raise ValueError(
            f"В migration.csv за {migration_year} год отсутствуют "
            f"{len(missing_pairs)} пар округов. "
            f"Примеры: {sorted(missing_pairs)[:10]}"
        )

    return flows


def get_available_migration_years(path):
    migration = pd.read_csv(path)

    migration["year"] = pd.to_numeric(
        migration["year"],
        errors="coerce",
    )

    return sorted(
        migration.loc[
            migration["year"].notna(),
            "year",
        ].astype(int).unique().tolist()
    )


def train_model(training_data):
    missing_features = (
        set(FEATURE_COLUMNS) - set(training_data.columns)
    )

    if missing_features:
        raise ValueError(
            "В migration_training.csv отсутствуют признаки: "
            f"{sorted(missing_features)}"
        )

    training_data = training_data.dropna(
        subset=["migration_share"]
    ).copy()

    X = training_data[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = pd.to_numeric(
        training_data["migration_share"],
        errors="coerce",
    )

    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=4,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(X, y)

    return model, training_data.loc[valid_rows].copy()


def create_pair_matrix(
    source_district,
    destination_district,
    source_subjects,
    destination_subjects,
    features_by_subject,
    reachability,
    district_flow,
    model,
):
    records = []

    for source in source_subjects:
        for destination in destination_subjects:
            record = {
                "source_subject": source,
                "destination_subject": destination,
                "reachability": float(
                    reachability.loc[source, destination]
                ),
            }

            for feature in DEMOGRAPHIC_FEATURES:
                record[f"source_{feature}"] = (
                    features_by_subject[source][feature]
                )
                record[f"destination_{feature}"] = (
                    features_by_subject[destination][feature]
                )

            records.append(record)

    pair_data = pd.DataFrame(records)

    X = pair_data[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    scores = model.predict(X)
    shares = normalize_scores(scores)

    pair_data["raw_score"] = scores
    pair_data["predicted_share_within_pair"] = shares
    pair_data["district_migration"] = district_flow
    pair_data["predicted_migration"] = shares * district_flow

    matrix = pair_data.pivot(
        index="source_subject",
        columns="destination_subject",
        values="predicted_migration",
    )

    matrix = matrix.loc[
        source_subjects,
        destination_subjects,
    ]

    expected_sum = float(district_flow)
    actual_sum = float(matrix.to_numpy().sum())

    if not np.isclose(
        actual_sum,
        expected_sum,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Сумма матрицы пары округов не совпадает "
            f"с заданным объёмом миграции: "
            f"{source_district} → {destination_district}; "
            f"ожидалось {expected_sum}, получено {actual_sum}."
        )

    return matrix, pair_data


def combine_pair_matrices(
    pair_matrices,
    subject_names,
    district_by_subject,
    year,
):
    matrix = pd.DataFrame(
        0.0,
        index=subject_names,
        columns=subject_names,
    )

    assigned_pairs = set()

    for (source_district, destination_district), pair_matrix in (
        pair_matrices.items()
    ):
        source_subjects = [
            subject
            for subject in subject_names
            if district_by_subject[subject] == source_district
        ]

        destination_subjects = [
            subject
            for subject in subject_names
            if district_by_subject[subject] == destination_district
        ]

        expected_index = set(source_subjects)
        expected_columns = set(destination_subjects)

        if (
            set(pair_matrix.index) != expected_index
            or set(pair_matrix.columns) != expected_columns
        ):
            raise ValueError(
                f"Размеры или состав матрицы пары округов "
                f"{source_district} → {destination_district} "
                f"не соответствуют составу субъектов."
            )

        if not np.isfinite(pair_matrix.to_numpy()).all():
            raise ValueError(
                f"Матрица {source_district} → "
                f"{destination_district} за {year} год "
                "содержит нечисловые или бесконечные значения."
            )

        if (pair_matrix.to_numpy() < 0).any():
            raise ValueError(
                f"Матрица {source_district} → "
                f"{destination_district} за {year} год "
                "содержит отрицательные значения."
            )

        pair_key = (source_district, destination_district)

        if pair_key in assigned_pairs:
            raise ValueError(
                f"Пара округов повторяется: {pair_key}"
            )

        assigned_pairs.add(pair_key)

        matrix.loc[
            source_subjects,
            destination_subjects,
        ] = pair_matrix.loc[
            source_subjects,
            destination_subjects,
        ].to_numpy()

    expected_district_pairs = {
        (source, destination)
        for source in DISTRICTS
        for destination in DISTRICTS
    }

    if assigned_pairs != expected_district_pairs:
        raise ValueError(
            "Набор обработанных пар федеральных округов "
            "не совпадает с ожидаемыми 64 парами."
        )

    if matrix.isna().any().any():
        raise ValueError(
            f"Итоговая матрица за {year} год содержит пропуски."
        )

    return matrix


def select_best_subject_records(
    indicators,
    subject_column,
    subject_names,
):
    indicators = indicators.copy()

    expected_subjects = {
        normalize_name(name): name
        for name in subject_names
    }

    indicators["_subject_key"] = (
        indicators[subject_column]
        .astype(str)
        .map(normalize_name)
    )

    indicators = indicators[
        indicators["_subject_key"].isin(
            expected_subjects
        )
    ].copy()

    indicators["_missing_count"] = (
        indicators.drop(
            columns=["_subject_key"]
        ).isna().sum(axis=1)
    )

    duplicate_mask = indicators[
        "_subject_key"
    ].duplicated(keep=False)

    duplicates = indicators[duplicate_mask]

    if not duplicates.empty:
        print(
            "Обнаружены повторные записи субъектов. "
            "Выбираются записи с минимальным количеством "
            "пропущенных значений:"
        )

        for subject_key, group in duplicates.groupby(
            "_subject_key",
            sort=True,
        ):
            best_missing = group[
                "_missing_count"
            ].min()

            best_rows = group[
                group["_missing_count"] == best_missing
            ]

            print(
                f"{expected_subjects[subject_key]}: "
                f"записей {len(group)}, "
                f"пропусков в выбранной записи "
                f"{best_missing}"
            )

            if len(best_rows) > 1:
                print(
                    "  Внимание: несколько записей имеют "
                    "одинаковое минимальное число пропусков. "
                    "Будет выбрана первая из них."
                )

    indicators = (
        indicators
        .sort_values(
            ["_subject_key", "_missing_count"],
            kind="stable",
        )
        .drop_duplicates(
            subset="_subject_key",
            keep="first",
        )
        .drop(
            columns=["_subject_key", "_missing_count"]
        )
        .reset_index(drop=True)
    )

    found_subjects = {
        normalize_name(name)
        for name in indicators[subject_column]
    }

    missing_subjects = (
        set(expected_subjects) - found_subjects
    )

    if missing_subjects:
        raise ValueError(
            "В indicators.csv отсутствуют субъекты "
            "из матрицы доступности:\n"
            + "\n".join(
                expected_subjects[key]
                for key in sorted(missing_subjects)
            )
        )

    print(
        "Количество записей субъектов после устранения "
        f"дубликатов: {len(indicators)}"
    )

    return indicators


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PAIR_MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_MATRICES_DIR.mkdir(parents=True, exist_ok=True)

    required_paths = [
        TRAINING_DATA_PATH,
        SUBJECT_INDICATORS_PATH,
        SUBJECT_REACHABILITY_PATH,
        DISTRICT_MIGRATION_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден необходимый файл: {path}"
            )

    print("Загрузка данных...")

    training_data = pd.read_csv(TRAINING_DATA_PATH)
    indicators = pd.read_csv(SUBJECT_INDICATORS_PATH)

    reachability = pd.read_csv(
        SUBJECT_REACHABILITY_PATH,
        index_col=0,
    )

    reachability.index = (
        reachability.index.astype(str).str.strip()
    )
    reachability.columns = (
        reachability.columns.astype(str).str.strip()
    )

    if reachability.index.duplicated().any():
        raise ValueError(
            "В матрице доступности повторяются названия субъектов "
            "в строках."
        )

    if reachability.columns.duplicated().any():
        raise ValueError(
            "В матрице доступности повторяются названия субъектов "
            "в столбцах."
        )

    if set(reachability.index) != set(reachability.columns):
        raise ValueError(
            "Состав субъектов в строках и столбцах матрицы "
            "доступности не совпадает."
        )

    subject_names = reachability.index.tolist()

    subject_column = find_subject_column(
        indicators,
        subject_names,
    )

    indicators = select_best_subject_records(
        indicators=indicators,
        subject_column=subject_column,
        subject_names=subject_names,
    )

    district_column = find_district_column(
        indicators,
        subject_column,
        subject_names,
    )

    reachability = reachability.loc[
        subject_names,
        subject_names,
    ].apply(pd.to_numeric, errors="coerce")

    if reachability.isna().any().any():
        raise ValueError(
            "Матрица доступности содержит пропущенные "
            "или нечисловые значения."
        )

    max_training_year = int(
        pd.to_numeric(
            training_data["feature_year"],
            errors="coerce",
        ).max()
    )

    feature_years = sorted(
        {
            year
            for year in range(1990, max_training_year + 1)
            if build_indicator_column_map(
                indicators.columns,
                year,
            ) is not None
        }
    )

    migration_years = get_available_migration_years(
        DISTRICT_MIGRATION_PATH
    )

    candidate_years = sorted(
        set(feature_years).intersection(migration_years)
    )

    if not candidate_years:
        raise ValueError(
            "Нет общих годов для демографических признаков "
            "и данных о миграции."
        )

    print(
        f"Годы, для которых найдены столбцы признаков: "
        f"{len(feature_years)}"
    )
    print(
        f"Годы миграции: "
        f"{min(migration_years)}–{max(migration_years)}"
    )
    print(
        f"Общие годы для обработки: "
        f"{min(candidate_years)}–{max(candidate_years)} "
        f"({len(candidate_years)} лет)"
    )

    print("Обучение случайного леса на всей исторической выборке...")

    model, fitted_training_data = train_model(training_data)

    training_years = sorted(
        pd.to_numeric(
            fitted_training_data["year"],
            errors="coerce",
        ).dropna().astype(int).unique().tolist()
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "training_years": training_years,
        },
        MODEL_PATH,
    )

    print(f"Модель сохранена: {MODEL_PATH}")

    all_manifest_records = []
    all_predictions = []
    processed_years = []
    skipped_years = []

    for year in candidate_years:
        print()
        print("=" * 70)
        print(f"Обработка {year} года")
        print("=" * 70)

        try:
            feature_year, column_map, subject_indicators = (
                select_feature_year(
                    indicators=indicators,
                    subject_column=subject_column,
                    subject_names=subject_names,
                    max_year=year,
                )
            )

            year_subject_names = subject_indicators[
                subject_column
            ].astype(str).tolist()

            year_subject_keys = {
                normalize_name(subject)
                for subject in year_subject_names
            }

            year_subject_names = [
                subject
                for subject in subject_names
                if normalize_name(subject) in year_subject_keys
            ]

            year_reachability = reachability.loc[
                year_subject_names,
                year_subject_names,
            ]

            district_flows = load_district_flows_for_year(
                DISTRICT_MIGRATION_PATH,
                year,
            )

            features_by_subject, district_by_subject = (
                build_subject_data(
                    subject_names=year_subject_names,
                    subject_indicators=subject_indicators,
                    subject_column=subject_column,
                    district_column=district_column,
                    column_map=column_map,
                    reachability=year_reachability,
                    indicators=indicators,
                )
            )

            year_pair_dir = PAIR_MATRICES_DIR / str(year)
            year_pair_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pair_matrices = {}
            year_manifest_records = []
            year_predictions = []

            pair_number = 0

            for source_district in DISTRICTS:
                source_subjects = [
                    subject
                    for subject in year_subject_names
                    if district_by_subject[subject]
                    == source_district
                ]

                for destination_district in DISTRICTS:
                    destination_subjects = [
                        subject
                        for subject in year_subject_names
                        if district_by_subject[subject]
                        == destination_district
                    ]

                    district_flow = district_flows[
                        (source_district, destination_district)
                    ]

                    pair_matrix, pair_data = create_pair_matrix(
                        source_district=source_district,
                        destination_district=destination_district,
                        source_subjects=source_subjects,
                        destination_subjects=destination_subjects,
                        features_by_subject=features_by_subject,
                        reachability=year_reachability,
                        district_flow=district_flow,
                        model=model,
                    )

                    pair_number += 1

                    filename = (
                        f"{pair_number:02d}_"
                        f"{normalize_name(source_district)}__"
                        f"{normalize_name(destination_district)}.csv"
                    )

                    matrix_path = year_pair_dir / filename

                    pair_matrix.to_csv(
                        matrix_path,
                        encoding="utf-8-sig",
                    )

                    pair_matrices[
                        (source_district, destination_district)
                    ] = pair_matrix

                    pair_data["source_federal_district"] = (
                        source_district
                    )
                    pair_data["destination_federal_district"] = (
                        destination_district
                    )
                    pair_data["feature_year"] = feature_year
                    pair_data["migration_year"] = year

                    year_predictions.append(pair_data)

                    year_manifest_records.append({
                        "feature_year": feature_year,
                        "migration_year": year,
                        "source_federal_district": source_district,
                        "destination_federal_district": (
                            destination_district
                        ),
                        "district_migration": district_flow,
                        "source_subject_count": len(source_subjects),
                        "destination_subject_count": len(
                            destination_subjects
                        ),
                        "matrix_sum": float(
                            pair_matrix.to_numpy().sum()
                        ),
                        "matrix_path": str(
                            matrix_path.relative_to(ROOT_DIR)
                        ),
                    })

                    print(
                        f"{pair_number:02d}/64: "
                        f"{source_district} → "
                        f"{destination_district}; "
                        f"объём = {district_flow:.2f}"
                    )

            combined_matrix = combine_pair_matrices(
                pair_matrices=pair_matrices,
                subject_names=year_subject_names,
                district_by_subject=district_by_subject,
                year=year,
            )

            combined_path = (
                COMBINED_MATRICES_DIR
                / f"subject_migration_matrix_{year}.csv"
            )

            combined_matrix.to_csv(
                combined_path,
                encoding="utf-8-sig",
            )

            manifest_total = sum(
                record["district_migration"]
                for record in year_manifest_records
            )

            matrix_total = float(
                combined_matrix.to_numpy().sum()
            )

            if not np.isclose(
                manifest_total,
                matrix_total,
                rtol=1e-9,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"Сумма итоговой матрицы за {year} год "
                    "не совпадает с суммой миграции между округами."
                )

            all_manifest_records.extend(
                year_manifest_records
            )
            all_predictions.extend(year_predictions)
            processed_years.append(year)

            print(
                f"Год {year} обработан: "
                f"64 матрицы пар округов, "
                f"итоговая матрица {len(combined_matrix)} × "
                f"{len(combined_matrix.columns)}."
            )
            print(
                f"Суммарная миграция: {matrix_total:.2f}"
            )
            print(f"Итоговая матрица: {combined_path}")

        except (ValueError, KeyError) as error:
            print(
                f"Год {year} пропущен из-за ошибки данных: {error}"
            )
            skipped_years.append({
                "year": year,
                "reason": str(error),
            })

    if not all_manifest_records:
        raise RuntimeError(
            "Не удалось построить ни одной годовой матрицы."
        )

    manifest = pd.DataFrame(all_manifest_records)
    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )
    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "model": "RandomForestRegressor",
        "parameters": {
            "n_estimators": 300,
            "max_depth": 5,
            "min_samples_leaf": 4,
            "max_features": 0.8,
            "random_state": RANDOM_STATE,
        },
        "training_years": training_years,
        "training_rows": int(len(fitted_training_data)),
        "feature_columns": FEATURE_COLUMNS,
        "subject_count": int(len(year_subject_names)),
        "district_count": len(DISTRICTS),
        "processed_years": processed_years,
        "skipped_years": skipped_years,
        "matrices_per_year": 64,
        "subject_pair_count": int(len(predictions)),
        "sum_of_district_flows": float(
            manifest["district_migration"].sum()
        ),
        "sum_of_subject_matrices": float(
            manifest["matrix_sum"].sum()
        ),
        "model_path": str(MODEL_PATH),
        "pair_matrices_dir": str(PAIR_MATRICES_DIR),
        "combined_matrices_dir": str(COMBINED_MATRICES_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "predictions_path": str(PREDICTIONS_PATH),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print()
    print("=" * 70)
    print("Обработка завершена.")
    print(f"Обработано лет: {len(processed_years)}")
    print(f"Пропущено лет: {len(skipped_years)}")
    print(f"Годы обработки: {processed_years}")
    print(f"Всего матриц пар округов: {len(manifest)}")
    print(f"Всего пар субъектов: {len(predictions)}")
    print(
        "Сумма миграции по матрицам округов: "
        f"{manifest['district_migration'].sum():.2f}"
    )
    print(
        "Сумма миграции по матрицам субъектов: "
        f"{manifest['matrix_sum'].sum():.2f}"
    )
    print(f"Промежуточные матрицы: {PAIR_MATRICES_DIR}")
    print(f"Итоговые матрицы: {COMBINED_MATRICES_DIR}")
    print(f"Реестр матриц: {MANIFEST_PATH}")
    print(f"Подробные прогнозы: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()