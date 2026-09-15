from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")


def print_dataframe_characteristics(path):
    print(f"\nФайл: {path}")

    excel_file = pd.ExcelFile(path)

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)

        print(f"\nЛист: {sheet}")
        print(f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
        print("Столбцы:")

        for column in df.columns:
            print(f"  {column}: {df[column].dtype}")

        print(f"Пропущенных значений: {df.isna().sum().sum()}")
        print(f"Дубликатов строк: {df.duplicated().sum()}")


def print_json_characteristics(path):
    import json

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"\nФайл: {path}")
    print(f"Тип данных: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"Ключи верхнего уровня: {list(data.keys())}")

        for key, value in data.items():
            if isinstance(value, list):
                print(f"{key}: список из {len(value)} элементов")
            elif isinstance(value, dict):
                print(f"{key}: объект с {len(value)} ключами")
            else:
                print(f"{key}: {type(value).__name__}")


def main():
    for path in sorted(RAW_DIR.rglob("*")):
        if path.name.startswith("~$"):
            continue

        if path.suffix.lower() in {".xlsx", ".xls"}:
            print_dataframe_characteristics(path)
        elif path.suffix.lower() == ".json":
            print_json_characteristics(path)


if __name__ == "__main__":
    main()