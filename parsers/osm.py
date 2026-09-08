from pathlib import Path

import requests

from load_manager import log_load, save_if_changed


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

QUERY = """
[out:json][timeout:300];

area["ISO3166-1"="RU"]->.searchArea;

relation["boundary"="administrative"]["admin_level"="4"](area.searchArea);

out geom;
"""


def query_overpass(session, query):
    response = session.post(
        OVERPASS_URL,
        data=query,
        headers={
            "User-Agent": "CourseworkDataCollector/1.0",
        },
        timeout=300,
    )

    response.raise_for_status()

    return response.content


def run():
    output_path = Path(
        "data/raw/osm/subjects_ru.json"
    )

    try:
        data = query_overpass(
            requests.Session(),
            QUERY,
        )

        changed = save_if_changed(
            data,
            output_path,
        )

        status = (
            "success"
            if changed
            else "skipped"
        )

        log_load(
            "OpenStreetMap",
            status,
            output_path,
            len(data),
        )

        print(
            f"OpenStreetMap: {status.upper()} "
            f"{len(data):,} bytes -> "
            f"{output_path}"
        )

    except Exception as error:
        log_load(
            "OpenStreetMap",
            "error",
            output_path,
            error=str(error),
        )

        raise