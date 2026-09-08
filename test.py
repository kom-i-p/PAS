import requests
from bs4 import BeautifulSoup


URL = "https://www.fedstat.ru/indicator/31556"
DOWNLOAD_URL = "https://www.fedstat.ru/indicator/31556/download"


def main():
    session = requests.Session()

    response = session.get(URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    token = soup.find("input", {"name": "token"})

    if token is None:
        raise RuntimeError("Токен не найден")

    token = token.get("value")

    data = {
        "struts.token.name": "token",
        "token": token,
        "id": "31556",
        "format": "excel",
    }

    response = session.post(DOWNLOAD_URL, data=data)
    response.raise_for_status()

    with open("fedstat_31556.xlsx", "wb") as file:
        file.write(response.content)

    print(f"Размер файла: {len(response.content):,} байт")
    print(f"Content-Type: {response.headers.get('Content-Type')}")


if __name__ == "__main__":
    main()