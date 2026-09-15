from parsers.fedstat import run as run_fedstat
from parsers.osm import run as run_osm
from parsers.rosstat import run as run_rosstat
from preprocessing import main as run_preprocessing
from analysis.characteristics import main as run_characteristics
from analysis.preview import main as run_preview


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

    run_analysis(
        "Preprocessing",
        run_preprocessing,
    )

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


if __name__ == "__main__":
    main()