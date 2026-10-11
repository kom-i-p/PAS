import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MIGRATION_PATH = PROCESSED_DIR / "migration.csv"
INDICATORS_PATH = PROCESSED_DIR / "district_indicators.csv"
REACHABILITY_PATH = PROCESSED_DIR / "district_reachability_matrix.csv"

OUTPUT_PATH = PROCESSED_DIR / "migration_training.csv"
REPORT_PATH = PROCESSED_DIR / "migration_training_report.json"

DISTRICT_MAPPING = {
    "Дальневосточный": "042 Дальневосточный федеральный округ",
    "Приволжский": "033 Приволжский федеральный округ",
    "Северо-Западный": "031 Северо-Западный федеральный округ",
    "Северо-Кавказский": "038 Северо-Кавказский федеральный округ",
    "Сибирский": "041 Сибирский федеральный округ",
    "Уральский": "034 Уральский федеральный округ",
    "Центральный": "030 Центральный федеральный округ",
    "Южный": "040 Южный федеральный округ",
}

FEATURE_GROUPS = {
    "population": "population",
    "births": "births",
    "deaths": "deaths",
    "urbanization": "urbanization",
    "population_change": "population_change",
    "population_growth": "population_growth",
    "natural_change": "natural_change",
    "birth_rate": "birth_rate",
    "death_rate": "death_rate",
}


def load_data():
    migration = pd.read_csv(MIGRATION_PATH)
    indicators = pd.read_csv(INDICATORS_PATH)
    reachability = pd.read_csv(REACHABILITY_PATH, index_col=0)

    migration["year"] = pd.to_numeric(
        migration["year"], errors="raise"
    ).astype(int)

    migration["migration"] = pd.to_numeric(
        migration["migration"], errors="raise"
    )

    indicators["federal_district"] = (
        indicators["federal_district"].astype(str).str.strip()
    )

    reachability.index = reachability.index.astype(str).str.strip()
    reachability.columns = reachability.columns.astype(str).str.strip()

    reachability = reachability.apply(
        pd.to_numeric, errors="coerce"
    )

    return migration, indicators, reachability


def get_population_column(columns, year):
    candidates = [
        str(year),
        f"{year}.0",
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def get_feature_column(columns, feature, year):
    column = f"{feature}_{year}"

    if column in columns:
        return column

    return None


def get_available_features(indicators, feature_year):
    columns = set(indicators.columns)
    feature_columns = {}

    population_column = get_population_column(
        columns, feature_year
    )

    if population_column is None:
        return None

    feature_columns["population"] = population_column

    for feature in FEATURE_GROUPS:
        if feature == "population":
            continue

        column = get_feature_column(
            columns, feature, feature_year
        )

        if column is None:
            return None

        feature_columns[feature] = column

    return feature_columns


def get_usable_years(migration, indicators):
    migration_years = sorted(migration["year"].unique())
    usable_years = []
    skipped_years = []

    indicators_by_district = indicators.set_index(
        "federal_district"
    )

    for target_year in migration_years:
        feature_year = target_year

        feature_columns = get_available_features(
            indicators, feature_year
        )

        if feature_columns is None:
            skipped_years.append({
                "migration_year": int(target_year),
                "feature_year": int(feature_year),
                "reason": "Недостаточно необходимых столбцов",
            })
            continue

        required_columns = list(feature_columns.values())

        missing_districts = [
            district
            for district in DISTRICT_MAPPING.values()
            if district not in indicators_by_district.index
        ]

        if missing_districts:
            raise ValueError(
                "В district_indicators.csv отсутствуют округа: "
                f"{missing_districts}"
            )

        feature_data = indicators_by_district.loc[
            list(DISTRICT_MAPPING.values()),
            required_columns,
        ]

        feature_data = feature_data.apply(
            pd.to_numeric, errors="coerce"
        )

        if feature_data.isna().any().any():
            skipped_years.append({
                "migration_year": int(target_year),
                "feature_year": int(feature_year),
                "reason": "Есть пропуски в демографических признаках",
            })
            continue

        year_data = migration[
            migration["year"] == target_year
        ]

        expected_pairs = len(DISTRICT_MAPPING) ** 2

        actual_pairs = year_data[
            ["source_federal_district", "destination_federal_district"]
        ].drop_duplicates().shape[0]

        if actual_pairs != expected_pairs or len(year_data) != expected_pairs:
            skipped_years.append({
                "migration_year": int(target_year),
                "feature_year": int(feature_year),
                "reason": (
                    "Неполная матрица миграции: "
                    f"строк {len(year_data)}, "
                    f"уникальных пар {actual_pairs}, "
                    f"ожидалось {expected_pairs}"
                ),
            })
            continue

        usable_years.append({
            "migration_year": int(target_year),
            "feature_year": int(feature_year),
            "feature_columns": feature_columns,
        })

    return usable_years, skipped_years


def validate_source_data(migration, indicators, reachability):
    required_migration_columns = {
        "year",
        "source_federal_district",
        "destination_federal_district",
        "migration",
    }

    missing_columns = (
        required_migration_columns - set(migration.columns)
    )

    if missing_columns:
        raise ValueError(
            f"В migration.csv отсутствуют столбцы: {missing_columns}"
        )

    if migration[
        [
            "year",
            "source_federal_district",
            "destination_federal_district",
            "migration",
        ]
    ].isna().any().any():
        raise ValueError(
            "В migration.csv найдены пропущенные значения"
        )

    if migration.duplicated(
        [
            "year",
            "source_federal_district",
            "destination_federal_district",
        ]
    ).any():
        raise ValueError(
            "В migration.csv найдены повторяющиеся пары за один год"
        )

    if (migration["migration"] < 0).any():
        raise ValueError(
            "В migration.csv обнаружены отрицательные значения "
            "миграции. Перед нормализацией необходимо выяснить, "
            "является ли показатель потоком или миграционным сальдо."
        )

    if indicators["federal_district"].duplicated().any():
        raise ValueError(
            "В district_indicators.csv есть повторяющиеся округа"
        )

    expected_districts = set(DISTRICT_MAPPING.values())

    indicator_districts = set(
        indicators["federal_district"]
    )

    if expected_districts != indicator_districts:
        raise ValueError(
            "Состав округов в district_indicators.csv "
            "не совпадает с ожидаемым.\n"
            f"Отсутствуют: {sorted(expected_districts - indicator_districts)}\n"
            f"Лишние: {sorted(indicator_districts - expected_districts)}"
        )

    if set(reachability.index) != expected_districts:
        raise ValueError(
            "Названия строк матрицы доступности "
            "не совпадают с названиями округов"
        )

    if set(reachability.columns) != expected_districts:
        raise ValueError(
            "Названия столбцов матрицы доступности "
            "не совпадают с названиями округов"
        )

    reachability = reachability.loc[
        list(DISTRICT_MAPPING.values()),
        list(DISTRICT_MAPPING.values()),
    ]

    if reachability.isna().any().any():
        raise ValueError(
            "В матрице доступности есть пропуски или нечисловые значения"
        )

    if not np.isfinite(reachability.to_numpy()).all():
        raise ValueError(
            "В матрице доступности есть бесконечные значения"
        )

    return reachability


def build_training_dataset(
    migration,
    indicators,
    reachability,
    usable_years,
):
    indicators_by_district = indicators.set_index(
        "federal_district"
    )

    migration["source_district"] = migration[
        "source_federal_district"
    ].map(DISTRICT_MAPPING)

    migration["destination_district"] = migration[
        "destination_federal_district"
    ].map(DISTRICT_MAPPING)

    if migration[
        ["source_district", "destination_district"]
    ].isna().any().any():
        unknown_sources = sorted(
            migration.loc[
                migration["source_district"].isna(),
                "source_federal_district",
            ].unique()
        )

        unknown_destinations = sorted(
            migration.loc[
                migration["destination_district"].isna(),
                "destination_federal_district",
            ].unique()
        )

        raise ValueError(
            "Не удалось сопоставить названия округов.\n"
            f"Источники: {unknown_sources}\n"
            f"Назначения: {unknown_destinations}"
        )

    usable_target_years = {
        item["migration_year"] for item in usable_years
    }

    migration = migration[
        migration["year"].isin(usable_target_years)
    ].copy()

    annual_totals = migration.groupby("year")[
        "migration"
    ].sum()

    rows = []

    for year_info in usable_years:
        target_year = year_info["migration_year"]
        feature_year = year_info["feature_year"]
        feature_columns = year_info["feature_columns"]

        year_migration = migration[
            migration["year"] == target_year
        ].copy()

        total_migration = float(
            annual_totals.loc[target_year]
        )

        year_migration["total_migration"] = total_migration

        if total_migration > 0:
            year_migration["migration_share"] = (
                year_migration["migration"] / total_migration
            )
        else:
            year_migration["migration_share"] = 0.0

        for feature, column in feature_columns.items():
            source_values = indicators_by_district[column]

            year_migration[f"source_{feature}"] = (
                year_migration["source_district"].map(source_values)
            )

            year_migration[f"destination_{feature}"] = (
                year_migration["destination_district"].map(
                    source_values
                )
            )

        year_migration["reachability"] = [
            reachability.loc[source, destination]
            for source, destination in zip(
                year_migration["source_district"],
                year_migration["destination_district"],
            )
        ]

        year_migration["feature_year"] = feature_year

        output_columns = [
            "year",
            "feature_year",
            "source_federal_district",
            "destination_federal_district",
            "source_district",
            "destination_district",
            "reachability",
            "total_migration",
            "migration",
            "migration_share",
        ]

        for feature in FEATURE_GROUPS:
            output_columns.extend([
                f"source_{feature}",
                f"destination_{feature}",
            ])

        rows.append(year_migration[output_columns])

    if not rows:
        raise ValueError(
            "Не удалось сформировать обучающую выборку: "
            "нет подходящих лет"
        )

    result = pd.concat(rows, ignore_index=True)

    return result


def build_report(
    migration,
    indicators,
    reachability,
    usable_years,
    skipped_years,
    training_data,
):
    year_details = []

    for year_info in usable_years:
        year = year_info["migration_year"]

        year_data = training_data[
            training_data["year"] == year
        ]

        total = float(year_data["migration"].sum())

        year_details.append({
            "migration_year": year,
            "feature_year": year_info["feature_year"],
            "matrix_rows": int(len(year_data)),
            "matrix_columns": int(
                year_data["source_district"].nunique()
            ),
            "total_migration": total,
            "share_sum": float(
                year_data["migration_share"].sum()
            ),
        })

    return {
        "source_shapes": {
            "migration": list(migration.shape),
            "district_indicators": list(indicators.shape),
            "district_reachability": list(reachability.shape),
        },
        "migration_source_years": sorted(
            int(year) for year in migration["year"].unique()
        ),
        "usable_years": [
            item["migration_year"] for item in usable_years
        ],
        "first_usable_year": (
            usable_years[0]["migration_year"]
            if usable_years else None
        ),
        "last_usable_year": (
            usable_years[-1]["migration_year"]
            if usable_years else None
        ),
        "last_feature_year": (
            usable_years[-1]["feature_year"]
            if usable_years else None
        ),
        "feature_groups": list(FEATURE_GROUPS),
        "district_mapping": DISTRICT_MAPPING,
        "skipped_years": skipped_years,
        "training_shape": list(training_data.shape),
        "training_missing_values": int(
            training_data.isna().sum().sum()
        ),
        "training_duplicate_rows": int(
            training_data.duplicated(
                [
                    "year",
                    "source_district",
                    "destination_district",
                ]
            ).sum()
        ),
        "year_details": year_details,
    }


def main():
    migration, indicators, reachability = load_data()

    reachability = validate_source_data(
        migration,
        indicators,
        reachability,
    )

    usable_years, skipped_years = get_usable_years(
        migration,
        indicators,
    )

    if not usable_years:
        raise ValueError(
            "Не найдено ни одного года, для которого "
            "есть миграционные данные и полные признаки."
        )

    training_data = build_training_dataset(
        migration,
        indicators,
        reachability,
        usable_years,
    )

    if training_data.isna().any().any():
        missing = training_data.isna().sum()
        missing = missing[missing > 0].to_dict()

        raise ValueError(
            f"В обучающей выборке обнаружены пропуски: {missing}"
        )

    if not np.isfinite(
        training_data.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError(
            "В обучающей выборке обнаружены бесконечные значения"
        )

    report = build_report(
        migration,
        indicators,
        reachability,
        usable_years,
        skipped_years,
        training_data,
    )

    training_data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("Подготовка обучающей выборки завершена.")
    print(f"Исходная таблица миграции: {MIGRATION_PATH}")
    print(f"Исходная таблица показателей: {INDICATORS_PATH}")
    print(f"Обучающая выборка: {OUTPUT_PATH}")
    print(f"Диагностический отчёт: {REPORT_PATH}")
    print()

    print("Годы миграционных данных:")
    print(report["migration_source_years"])
    print()

    print("Годы, включённые в обучающую выборку:")
    print(report["usable_years"])
    print()

    print(
        "Последний год миграции в выборке:",
        report["last_usable_year"],
    )
    print(
        "Последний год демографических признаков:",
        report["last_feature_year"],
    )
    print()

    print("Размер обучающей выборки:", training_data.shape)
    print(
        "Количество пропусков:",
        report["training_missing_values"],
    )
    print(
        "Количество дубликатов:",
        report["training_duplicate_rows"],
    )
    print()

    print("Проверка годовых матриц:")

    for item in report["year_details"]:
        print(
            f"Миграция за {item['migration_year']}, "
            f"признаки за {item['feature_year']}: "
            f"{item['matrix_rows']} строк, "
            f"сумма миграции = {item['total_migration']:.2f}, "
            f"сумма долей = {item['share_sum']:.8f}"
        )

    if skipped_years:
        print()
        print("Исключённые годы:")

        for item in skipped_years:
            print(
                f"{item['migration_year']}: {item['reason']}"
            )

    print()
    print("Первые строки обучающей выборки:")
    print(training_data.head().to_string(index=False))


if __name__ == "__main__":
    main()