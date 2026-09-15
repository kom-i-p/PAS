from pathlib import Path
import json
import re

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


FEDSTAT_FILES = {
    "population": "population_31556.xls",
    "births": "births_31606.xls",
    "deaths": "deaths_31617.xls",
    "urban_share": "urban_share_36057.xls",
}

MIGRATION_FILE = "migration.xlsx"

FEDERAL_DISTRICTS = {
    "Центральный федеральный округ",
    "Северо-Западный федеральный округ",
    "Южный федеральный округ",
    "Северо-Кавказский федеральный округ",
    "Приволжский федеральный округ",
    "Уральский федеральный округ",
    "Сибирский федеральный округ",
    "Дальневосточный федеральный округ",
}


def clean_text(value):
    if pd.isna(value):
        return value

    return str(value).strip()


def split_subject(value):
    value = clean_text(value)

    if pd.isna(value):
        return None, None

    parts = value.split(maxsplit=1)

    if len(parts) == 1:
        return parts[0], ""

    return parts[0], parts[1]


def preprocess_population(path):
    df = pd.read_excel(path, header=None)

    years = df.iloc[2].tolist()[4:]

    df = df.iloc[3:].reset_index(drop=True)

    territory_column = df.columns[0]

    df = df[df[territory_column].notna()].copy()

    df["_indent"] = (
        df[territory_column].astype(str).str.len()
        - df[territory_column].astype(str).str.lstrip().str.len()
    )

    df["_territory"] = (
        df[territory_column]
        .astype(str)
        .str.strip()
        .str.strip("'")
    )

    df = df[df["_indent"].isin([4, 8, 12])].copy()

    current_district = None
    rows = []

    for _, row in df.iterrows():
        indent = row["_indent"]
        territory = row["_territory"]

        if indent == 4:
            if "федеральный округ" in territory.lower():
                current_district = territory

            continue

        if indent in (8, 12):
            row["federal_district"] = current_district
            row["is_nested"] = indent == 12
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Не удалось найти территории в population")

    settlement_column = df.columns[3]

    df = df[
        df[settlement_column]
        .astype(str)
        .str.strip()
        .str.strip("'")
        .eq("w2:p_mest:11 все население")
    ].copy()

    code_name = df["_territory"].apply(split_subject)

    df["okato"] = code_name.map(lambda value: value[0])
    df["region"] = code_name.map(lambda value: value[1])

    value_columns = list(df.columns[4:4 + len(years)])

    result = df[
        [
            "okato",
            "region",
            "federal_district",
            "is_nested",
        ] + value_columns
    ].copy()

    result = result.rename(
        columns=dict(zip(value_columns, years))
    )

    return result.reset_index(drop=True)


def preprocess_fedstat(path, filter_sex=True):
    df = pd.read_excel(
        path,
        header=None,
    )

    years = df.iloc[2, 5:].tolist()

    df = df.iloc[3:].reset_index(drop=True)

    territory_column = df.columns[0]

    df = df[
        df[territory_column].notna()
    ].copy()

    df["_indent"] = (
        df[territory_column]
        .astype(str)
        .str.len()
        - df[territory_column]
        .astype(str)
        .str.lstrip()
        .str.len()
    )

    df["_territory"] = (
        df[territory_column]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["_indent"].isin([4, 8, 12])
    ].copy()

    current_district = None
    rows = []

    for _, row in df.iterrows():
        indent = row["_indent"]

        if indent == 4:
            current_district = row["_territory"]
            continue

        if indent in (8, 12):
            row["federal_district"] = current_district
            row["is_nested"] = indent == 12
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"Не найдены территории в файле {path}"
        )

    if filter_sex:
        sex_column = df.columns[3]
        settlement_column = df.columns[4]

        df = df[
            df[sex_column]
            .astype(str)
            .str.strip()
            .eq("12 Оба пола")
            & df[settlement_column]
            .astype(str)
            .str.strip()
            .eq(
                "w2:p_mest:11 все население"
            )
        ].copy()

    code_name = df["_territory"].apply(
        split_subject
    )

    df["okato"] = code_name.map(
        lambda x: x[0]
    )

    df["region"] = code_name.map(
        lambda x: x[1]
    )

    year_columns = []

    for column, year in zip(
        df.columns[5:5 + len(years)],
        years,
    ):
        if pd.isna(year):
            continue

        year = int(year)
        year_columns.append(
            (column, str(year))
        )

    result = df[
        [
            "okato",
            "region",
            "federal_district",
            "is_nested",
        ]
        + [column for column, _ in year_columns]
    ].copy()

    result = result.rename(
        columns=dict(year_columns)
    )

    result = result.reset_index(drop=True)

    return result


def find_migration_tables(path):
    excel_file = pd.ExcelFile(path)
    tables = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None,
        )

        for row in range(len(df)):
            for column in range(df.shape[1]):
                value = df.iat[row, column]

                if str(value).strip() != "Из федеральных округов":
                    continue

                year_row = row - 4
                year_column = column + 1

                if year_row < 0:
                    continue

                if year_column >= df.shape[1]:
                    continue

                year_value = df.iat[
                    year_row,
                    year_column,
                ]

                match = re.fullmatch(
                    r"\s*(\d{4})г\.\s*",
                    str(year_value),
                )

                if match is None:
                    continue

                tables.append(
                    (
                        sheet_name,
                        int(match.group(1)),
                        df,
                        row,
                        column,
                    )
                )

    return tables


def preprocess_migration(path):
    tables = find_migration_tables(path)

    result = []

    for sheet_name, year, df, start_row, start_column in tables:
        district_column = start_column

        first_value_column = start_column + 2

        districts = []

        for row in range(
            start_row + 1,
            start_row + 9,
        ):
            value = df.iat[row, district_column]

            if pd.isna(value):
                continue

            districts.append(
                str(value).strip()
            )

        if len(districts) != 8:
            continue

        destination_columns = range(
            first_value_column,
            first_value_column + 8,
        )

        for row, source_district in zip(
            range(start_row + 1, start_row + 9),
            districts,
        ):
            for column, destination_district in zip(
                destination_columns,
                districts,
            ):
                value = df.iat[row, column]

                result.append(
                    {
                        "year": year,
                        "source_federal_district":
                            source_district,
                        "destination_federal_district":
                            destination_district,
                        "migration": value,
                    }
                )

    result = pd.DataFrame(result)

    if result.empty:
        raise ValueError(
            "Не найдено ни одной матрицы миграции"
        )

    return result


def preprocess_osm(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    rows = []

    for element in data.get("elements", []):
        tags = element.get("tags", {})

        rows.append(
            {
                "osm_id": element.get("id"),
                "iso_code": tags.get("ISO3166-2"),
                "okato": tags.get("ssrf:code"),
                "name": tags.get("name"),
                "ref": tags.get("ref"),
                "population": tags.get("population"),
                "sqkm": tags.get("sqkm"),
                "admin_level": tags.get("admin_level"),
                "boundary": tags.get("boundary"),
                "geometry": json.dumps(
                    element.get("members", []),
                    ensure_ascii=False,
                ),
            }
        )

    return pd.DataFrame(rows)


def save_table(df, name):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = PROCESSED_DIR / f"{name}.csv"

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Processed: {len(df):,} rows -> {path}"
    )


def merge_fedstat_data(
    population,
    births,
    deaths,
    urban_share,
):
    keys = [
        "okato",
        "region",
        "federal_district",
        "is_nested",
    ]

    result = population.copy()

    for name, df in [
        ("births", births),
        ("deaths", deaths),
        ("urban_share", urban_share),
    ]:
        value_columns = [
            column
            for column in df.columns
            if column not in keys
        ]

        df = df[
            keys + value_columns
        ].copy()

        df = df.rename(
            columns={
                column: f"{name}_{column}"
                for column in value_columns
            }
        )

        result = result.merge(
            df,
            on=keys,
            how="left",
        )

    return result


from difflib import SequenceMatcher


def normalize_region_name(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = value.replace("ё", "е")

    value = re.sub(r"[«»\"'`]", "", value)
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"\s+", " ", value)

    value = value.replace("г. ", "")
    value = value.replace("г ", "")

    return value.strip()


def find_best_region_match(
    region,
    osm_regions,
    threshold=0.5,
):
    normalized_region = normalize_region_name(region)

    if not normalized_region:
        return None

    best_name = None
    best_score = 0

    for osm_region in osm_regions:
        normalized_osm = normalize_region_name(osm_region)

        score = SequenceMatcher(
            None,
            normalized_region,
            normalized_osm,
        ).ratio()

        if score > best_score:
            best_score = score
            best_name = osm_region

    if best_score >= threshold:
        return best_name

    return None


def merge_fedstat_osm(
    population,
    births,
    deaths,
    urban_share,
    osm,
    mapping,
):
    keys = [
        "region",
        "federal_district",
        "is_nested",
    ]

    result = population.copy()

    for name, table in [
        ("births", births),
        ("deaths", deaths),
        ("urban_share", urban_share),
    ]:
        value_columns = [
            column
            for column in table.columns
            if column not in [
                "okato",
                "region",
                "federal_district",
                "is_nested",
            ]
        ]

        table = table[
            keys + value_columns
        ].copy()

        table = table.rename(
            columns={
                column: f"{name}_{column}"
                for column in value_columns
            }
        )

        result = result.merge(
            table,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    result = result.merge(
        mapping,
        left_on="region",
        right_on="fedstat_name",
        how="left",
        validate="many_to_one",
    )

    osm = osm.drop_duplicates("name")

    result = result.merge(
        osm,
        left_on="osm_name",
        right_on="name",
        how="left",
        validate="many_to_one",
    )

    result = result.drop(
        columns=["fedstat_name", "name"],
        errors="ignore",
    )

    return result


def load_region_mapping(path):
    mapping = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    mapping = mapping.dropna(subset=["fedstat_name"])
    mapping["fedstat_name"] = (
        mapping["fedstat_name"]
        .astype(str)
        .str.strip()
    )

    mapping["osm_name"] = (
        mapping["osm_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    mapping = mapping.drop_duplicates("fedstat_name")

    return mapping


def main():
    population = preprocess_population(
        RAW_DIR / "fedstat" / FEDSTAT_FILES["population"]
    )
    save_table(population, "population")

    births = preprocess_fedstat(
        RAW_DIR / "fedstat" / FEDSTAT_FILES["births"]
    )
    save_table(births, "births")

    deaths = preprocess_fedstat(
        RAW_DIR / "fedstat" / FEDSTAT_FILES["deaths"]
    )
    save_table(deaths, "deaths")

    urban_share = preprocess_fedstat(
        RAW_DIR
        / "fedstat"
        / FEDSTAT_FILES["urban_share"],
        filter_sex=False,
    )
    save_table(
        urban_share,
        "urban_share",
    )

    migration = preprocess_migration(
        RAW_DIR
        / "rosstat"
        / MIGRATION_FILE
    )
    save_table(
        migration,
        "migration",
    )

    osm = preprocess_osm(
        RAW_DIR
        / "osm"
        / "subjects_ru.json"
    )
    save_table(osm, "osm_subjects")

    fedstat = merge_fedstat_data(
        population,
        births,
        deaths,
        urban_share,
    )

    save_table(
        fedstat,
        "fedstat",
    )

    mapping = load_region_mapping(
        PROCESSED_DIR / "region_name_mapping.csv"
    )

    combined = merge_fedstat_osm(
        population,
        births,
        deaths,
        urban_share,
        osm,
        mapping,
    )

    save_table(combined, "analytical_mart")


if __name__ == "__main__":
    main()