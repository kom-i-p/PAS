from parsers.fedstat import run as run_fedstat
from parsers.osm import run as run_osm
from parsers.rosstat import run as run_rosstat
from preprocessing import main as run_preprocessing
from analysis.characteristics import main as run_characteristics
from analysis.preview import main as run_preview
from analysis.eda import main as run_eda
from analysis.indicators import main as run_indicators
from analysis.subject_matrix import main as run_subject_matrix
from analysis.district_matrix import main as run_district_matrix


def run_parser(name, parser):
    print(f"=== {name} ===")

    try:
        parser()
        print(f"{name}: OK")
    except Exception as error:
        print(f"{name}: ERROR: {error}")


def run_analysis(name, analysis):
    print(f"=== {name} ===")

    try:
        analysis()
        print(f"{name}: OK")
    except Exception as error:
        print(f"{name}: ERROR: {error}")


def main():
    run_parser(
        "Rosstat",
        run_rosstat,
    )

    print()

    run_parser(
        "Fedstat",
        run_fedstat,
    )

    print()

    run_parser(
        "OpenStreetMap",
        run_osm,
    )

    print()

    print()

    run_analysis(
        "Characteristics",
        run_characteristics,
    )

    print()

    run_analysis(
        "Preview",
        run_preview,
    )

    run_analysis(
        "Preprocessing",
        run_preprocessing,
    )

    run_analysis(
        "EDA",
        run_eda,
    )

    print()

    run_analysis(
        "Indicators",
        run_indicators,
    )

    print()

    run_analysis("Subject matrix", run_subject_matrix)
    print()

    run_analysis("District matrix", run_district_matrix)
    print()


if __name__ == "__main__":
    main()