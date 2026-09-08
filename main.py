from parsers.fedstat import run as run_fedstat
from parsers.osm import run as run_osm
from parsers.rosstat import run as run_rosstat


def run_parser(name, parser):
    print(f"=== {name} ===")

    try:
        parser()
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


if __name__ == "__main__":
    main()