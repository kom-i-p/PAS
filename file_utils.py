OLE2_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
XLSX_SIGNATURE = b"PK\x03\x04"


def get_excel_extension(data):
    if data.startswith(OLE2_SIGNATURE):
        return ".xls"

    if data.startswith(XLSX_SIGNATURE):
        return ".xlsx"

    raise ValueError(
        "Downloaded file is not a valid Excel file"
    )