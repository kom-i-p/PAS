from pathlib import Path

import requests

from file_utils import get_excel_extension
from load_manager import log_load, save_if_changed


ROSTAT_FILES = {
    "migration": (
        "https://rosstat.gov.ru/storage/mediabank/"
        "Vnutriros_migr.xlsx"
    ),
}


def download_file(session, url):
    response = session.get(
        url,
        verify=False,
    )
    response.raise_for_status()

    return response.content


def run():
    output_dir = Path("data/raw/rosstat")

    with requests.Session() as session:
        for name, url in ROSTAT_FILES.items():
            output_path = None

            try:
                data = download_file(
                    session,
                    url,
                )

                extension = get_excel_extension(data)

                output_path = (
                    output_dir
                    / f"{name}{extension}"
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
                    "Rosstat",
                    status,
                    output_path,
                    len(data),
                )

                print(
                    f"Rosstat: {status.upper()} "
                    f"{len(data):,} bytes -> "
                    f"{output_path}"
                )

            except Exception as error:
                log_load(
                    "Rosstat",
                    "error",
                    output_path,
                    error=str(error),
                )

                raise