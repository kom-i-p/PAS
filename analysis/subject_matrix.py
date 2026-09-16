import json
from pathlib import Path

import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "osm" / "subjects_ru.json"
OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "subject_reachability_matrix.csv"
)


def load_osm_data():
    with RAW_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_geometries(data):
    geometries = {}

    for element in data["elements"]:
        if element.get("type") != "relation":
            continue

        tags = element.get("tags", {})
        name = tags.get("name")

        if not name:
            continue

        lines = []

        for member in element.get("members", []):
            geometry = member.get("geometry")

            if not geometry or len(geometry) < 2:
                continue

            points = [
                (point["lon"], point["lat"])
                for point in geometry
            ]

            lines.append(LineString(points))

        if not lines:
            continue

        geometries[name] = unary_union(lines)

    return geometries


def build_adjacency_graph(geometries):
    names = list(geometries)
    graph = {
        name: set()
        for name in names
    }

    for index, first_name in enumerate(names):
        first_geometry = geometries[first_name]

        for second_name in names[index + 1:]:
            second_geometry = geometries[second_name]

            if first_geometry.intersects(second_geometry):
                graph[first_name].add(second_name)
                graph[second_name].add(first_name)

    return graph


def calculate_distances(graph, start):
    distances = {start: 0}
    queue = [start]

    while queue:
        current = queue.pop(0)

        for neighbor in graph[current]:
            if neighbor in distances:
                continue

            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    return distances


def build_matrix(graph):
    names = list(graph)
    matrix = []

    for start in names:
        distances = calculate_distances(graph, start)

        row = [
            distances.get(name, float("nan"))
            for name in names
        ]

        matrix.append(row)

    return pd.DataFrame(
        matrix,
        index=names,
        columns=names,
    )


def save_matrix(matrix):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
    )

    print(
        f"Subject matrix: "
        f"{len(matrix):,} x {len(matrix.columns):,} "
        f"-> {OUTPUT_FILE}"
    )


def main():
    data = load_osm_data()
    geometries = build_geometries(data)

    print(f"Subjects with geometry: {len(geometries):,}")

    graph = build_adjacency_graph(geometries)

    edges = sum(
        len(neighbors)
        for neighbors in graph.values()
    ) // 2

    print(f"Adjacency edges: {edges:,}")

    matrix = build_matrix(graph)

    save_matrix(matrix)


if __name__ == "__main__":
    main()