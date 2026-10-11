import json
from pathlib import Path

import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "osm"
    / "subjects_ru.json"
)

ANALYTICAL_MART_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "analytical_mart.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "subject_reachability_matrix.csv"
)


MANUAL_DISTANCES = {
    ("Калининградская область", "Псковская область"): 3,
    ("Калининградская область", "Смоленская область"): 3,
}


def load_osm_data():
    with RAW_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_analytical_mart():
    return pd.read_csv(
        ANALYTICAL_MART_FILE,
        encoding="utf-8-sig",
    )


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


def filter_matched_geometries(geometries, analytical_mart):
    matched_names = set(
        analytical_mart["osm_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    filtered_geometries = {
        name: geometry
        for name, geometry in geometries.items()
        if name in matched_names
    }

    return filtered_geometries


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
    matrix = pd.DataFrame(
        index=names,
        columns=names,
        dtype=float,
    )

    for start in names:
        distances = calculate_distances(graph, start)

        for name in names:
            matrix.loc[start, name] = distances.get(
                name,
                float("nan"),
            )

    for (first, second), distance in MANUAL_DISTANCES.items():
        if first not in graph:
            raise ValueError(
                f"Субъект отсутствует в графе: {first}"
            )

        if second not in graph:
            raise ValueError(
                f"Субъект отсутствует в графе: {second}"
            )

        matrix.loc[first, second] = distance
        matrix.loc[second, first] = distance

    kaliningrad = "Калининградская область"

    if kaliningrad in graph:
        anchors = [
            name
            for name in (
                "Псковская область",
                "Смоленская область",
            )
            if name in graph
        ]

        for destination in names:
            if destination == kaliningrad:
                matrix.loc[kaliningrad, destination] = 0
                continue

            candidate_distances = []

            for anchor in anchors:
                anchor_distance = matrix.loc[
                    kaliningrad,
                    anchor,
                ]

                path_distance = matrix.loc[
                    anchor,
                    destination,
                ]

                if pd.notna(path_distance):
                    candidate_distances.append(
                        anchor_distance + path_distance
                    )

            if candidate_distances:
                distance = min(candidate_distances)

                matrix.loc[
                    kaliningrad,
                    destination,
                ] = distance

                matrix.loc[
                    destination,
                    kaliningrad,
                ] = distance

    return matrix


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
    analytical_mart = load_analytical_mart()

    geometries = build_geometries(data)

    print(
        f"Subjects with geometry: "
        f"{len(geometries):,}"
    )

    filtered_geometries = filter_matched_geometries(
        geometries,
        analytical_mart,
    )

    print(
        f"Subjects matched with analytical mart: "
        f"{len(filtered_geometries):,}"
    )

    print(
        f"Subjects excluded: "
        f"{len(geometries) - len(filtered_geometries):,}"
    )

    graph = build_adjacency_graph(filtered_geometries)

    edges = sum(
        len(neighbors)
        for neighbors in graph.values()
    ) // 2

    print(f"Adjacency edges: {edges:,}")

    matrix = build_matrix(graph)

    save_matrix(matrix)


if __name__ == "__main__":
    main()