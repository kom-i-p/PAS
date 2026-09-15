from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")


def print_dataframe_preview(path):
    print(f"\nФайл: {path}")

    excel_file = pd.ExcelFile(path)

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)

        print(f"\nЛист: {sheet}")
        print(df.head())


def print_json_structure(data, indent=0):
    prefix = " " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}: {{")
                print_json_structure(value, indent + 2)
                print(f"{prefix}}}")
            elif isinstance(value, list):
                print(f"{prefix}{key}: [{len(value)} элементов]")
                if value:
                    print_json_structure(value[0], indent + 2)
            else:
                print(f"{prefix}{key}: ...")

    elif isinstance(data, list):
        if data:
            print_json_structure(data[0], indent)
        else:
            print(f"{prefix}...")


def print_json_preview(path):
    import json

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"\nФайл: {path}")
    print_json_structure(data)


def main():
    for path in sorted(RAW_DIR.rglob("*")):
        if path.name.startswith("~$"):
            continue

        if path.suffix.lower() in {".xlsx", ".xls"}:
            print_dataframe_preview(path)
        elif path.suffix.lower() == ".json":
            print_json_preview(path)


if __name__ == "__main__":
    main()