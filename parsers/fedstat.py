from pathlib import Path

import requests
from bs4 import BeautifulSoup

from load_manager import log_load, save_if_changed

from file_utils import get_excel_extension


BASE_URL = "https://www.fedstat.ru"

INDICATORS = {
    "population": 31556,
    "urban_share": 36057,
    "deaths": 31617,
    "births": 31606,
}


def download_indicator(
    session,
    indicator_id,
):
    url = f"{BASE_URL}/indicator/{indicator_id}"

    response = session.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    token = soup.find(
        "input",
        {"name": "token"},
    )

    if token is None:
        raise RuntimeError(
            f"Token not found for indicator "
            f"{indicator_id}"
        )

    download_url = (
        f"{BASE_URL}/indicator/{indicator_id}/download"
    )

    data = {
        "struts.token.name": "token",
        "token": token["value"],
        "id": str(indicator_id),
        "format": "excel",
    }

    response = session.post(
        download_url,
        data=data,
    )
    response.raise_for_status()

    return response.content


def run():
    output_dir = Path("data/raw/fedstat")

    with requests.Session() as session:
        for name, indicator_id in INDICATORS.items():
            output_path = None

            try:
                data = download_indicator(
                    session,
                    indicator_id,
                )

                extension = get_excel_extension(data)

                output_path = (
                    output_dir
                    / f"{name}_{indicator_id}{extension}"
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
                    "Fedstat",
                    status,
                    output_path,
                    len(data),
                )

                print(
                    f"Fedstat: {status.upper()} "
                    f"{len(data):,} bytes -> "
                    f"{output_path}"
                )

            except Exception as error:
                log_load(
                    "Fedstat",
                    "error",
                    output_path,
                    error=str(error),
                )

                raise