from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
SUBJECT_MATRIX_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "subject_reachability_matrix.csv"
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
    / "district_reachability_matrix.csv"
)


def load_data():
    matrix = pd.read_csv(
        SUBJECT_MATRIX_FILE,
        index_col=0,
        encoding="utf-8-sig",
    )

    regions = pd.read_csv(
        ANALYTICAL_MART_FILE,
        encoding="utf-8-sig",
    )

    regions = regions[
        ["region", "federal_district"]
    ].dropna()

    regions = regions.drop_duplicates("region")

    return matrix, regions


def build_district_graph(matrix, regions):
    subject_to_district = dict(
        zip(
            regions["region"],
            regions["federal_district"],
        )
    )

    graph = {
        district: set()
        for district in regions["federal_district"]
    }

    for subject_a in matrix.index:
        district_a = subject_to_district.get(subject_a)

        if district_a is None:
            continue

        for subject_b in matrix.columns:
            district_b = subject_to_district.get(subject_b)

            if district_b is None or district_a == district_b:
                continue

            distance = matrix.loc[subject_a, subject_b]

            if distance == 1:
                graph[district_a].add(district_b)

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
    districts = list(graph)
    matrix = []

    for start in districts:
        distances = calculate_distances(
            graph,
            start,
        )

        row = [
            distances.get(
                district,
                float("nan"),
            )
            for district in districts
        ]

        matrix.append(row)

    return pd.DataFrame(
        matrix,
        index=districts,
        columns=districts,
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
        f"District matrix: "
        f"{len(matrix):,} x {len(matrix.columns):,} "
        f"-> {OUTPUT_FILE}"
    )


def main():
    matrix, regions = load_data()

    graph = build_district_graph(
        matrix,
        regions,
    )

    edges = sum(
        len(neighbors)
        for neighbors in graph.values()
    ) // 2

    print(f"Federal districts: {len(graph):,}")
    print(f"Adjacency edges: {edges:,}")

    result = build_matrix(graph)

    save_matrix(result)


if __name__ == "__main__":
    main()