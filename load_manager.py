from datetime import datetime
from hashlib import sha256
from pathlib import Path
import csv


LOG_PATH = Path("data/load_log.csv")


def get_hash(data):
    return sha256(data).hexdigest()


def get_file_hash(path):
    if not path.exists():
        return None

    return get_hash(path.read_bytes())


def save_file(data, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temp_path.write_bytes(data)
    temp_path.replace(output_path)


def log_load(
    source,
    status,
    path,
    size=0,
    error="",
):
    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = LOG_PATH.exists()

    with LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "datetime",
                    "source",
                    "status",
                    "path",
                    "size",
                    "error",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                source,
                status,
                path,
                size,
                error,
            ]
        )


def save_if_changed(
    data,
    output_path,
):
    new_hash = get_hash(data)
    old_hash = get_file_hash(output_path)

    if old_hash == new_hash:
        return False

    save_file(
        data,
        output_path,
    )

    return True