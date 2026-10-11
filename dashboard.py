from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
INDICATORS_FILE = PROCESSED_DIR / "indicators.csv"
FORECAST_FILE = PROCESSED_DIR / "population_forecast.csv"

INDICATORS = {
    "Численность населения": "population",
    "Изменение численности населения": "population_change",
    "Темп прироста/убыли населения, %": "population_growth",
    "Родившиеся": "births",
    "Умершие": "deaths",
    "Естественный прирост/убыль": "natural_change",
    "Коэффициент рождаемости": "birth_rate",
    "Коэффициент смертности": "death_rate",
    "Доля городского населения, %": "urbanization",
}


def normalize_name(value):
    value = str(value).strip().lower().replace("ё", "е")
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-zа-я0-9]", "", value)
    for suffix in ("автономныйокруг", "автономнаяобласть", "область",
                   "республика", "край", "городфедеральногозначения"):
        if value.startswith(suffix):
            value = value[len(suffix):]
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    return value


def read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def population_year_columns(df):
    result = []
    for column in df.columns:
        try:
            year = int(float(str(column)))
        except (TypeError, ValueError):
            continue
        if 1900 <= year <= 2100 and pd.to_numeric(
                df[column], errors="coerce"
        ).notna().any():
            result.append((year, column))
    return sorted(result)


def find_year_column(df, indicator, year):
    if indicator == "population":
        for column in df.columns:
            try:
                if int(float(str(column))) == int(year):
                    return column
            except (TypeError, ValueError):
                pass
        return None

    candidates = [
        f"{indicator}_{year}",
        f"{indicator}_{year}.0",
        f"{indicator}{year}",
        f"{year}_{indicator}",
        f"{year}{indicator}",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    normalized_indicator = re.sub(r"[^a-zа-я0-9]", "", indicator.lower())
    normalized_year = str(int(year))
    for column in df.columns:
        normalized_column = re.sub(
            r"[^a-zа-я0-9]", "", str(column).lower().replace("ё", "е")
        )
        if normalized_column in {
            normalized_indicator + normalized_year,
            normalized_indicator + normalized_year + "0",
            normalized_year + normalized_indicator,
            normalized_year + "0" + normalized_indicator,
        }:
            return column

    return None


def indicator_values(df, indicator, year):
    column = find_year_column(df, indicator, year)
    if column is not None:
        return pd.to_numeric(df[column], errors="coerce")

    if indicator in {"population_change", "population_growth"}:
        current_col = find_year_column(df, "population", year)
        previous_col = find_year_column(df, "population", year - 1)
        if current_col is None or previous_col is None:
            return pd.Series(np.nan, index=df.index)
        current = pd.to_numeric(df[current_col], errors="coerce")
        previous = pd.to_numeric(df[previous_col], errors="coerce")
        change = current - previous
        if indicator == "population_change":
            return change
        return (change / previous.replace(0, np.nan)) * 100

    return pd.Series(np.nan, index=df.index)


def _same_point(a, b, tolerance=1e-7):
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def _join_outer_rings(ways):
    """Собирает замкнутые кольца из сегментов outer, заданных OSM-координатами."""
    segments = []
    for way in ways:
        points = way.get("geometry") or []
        coords = []
        for point in points:
            try:
                coords.append([float(point["lon"]), float(point["lat"])])
            except (KeyError, TypeError, ValueError):
                continue
        if len(coords) >= 2:
            segments.append(coords)

    rings = []
    while segments:
        ring = segments.pop(0)
        changed = True
        while changed and not _same_point(ring[0], ring[-1]):
            changed = False
            for index, segment in enumerate(segments):
                if _same_point(ring[-1], segment[0]):
                    ring.extend(segment[1:])
                elif _same_point(ring[-1], segment[-1]):
                    ring.extend(reversed(segment[:-1]))
                elif _same_point(ring[0], segment[-1]):
                    ring = segment[:-1] + ring
                elif _same_point(ring[0], segment[0]):
                    ring = list(reversed(segment[1:])) + ring
                else:
                    continue
                segments.pop(index)
                changed = True
                break
        if len(ring) >= 4 and _same_point(ring[0], ring[-1]):
            ring[-1] = ring[0]
            rings.append(ring)
    return rings


def _fix_winding_order(geometry):
    """Исправляет порядок обхода колец (winding order) для GeoJSON.
    Внешние кольца должны быть против часовой стрелки (CCW), внутренние - по часовой (CW).
    Корректно обрабатывает полигоны, пересекающие 180-й меридиан (Чукотка).
    """
    if not isinstance(geometry, dict) or "coordinates" not in geometry:
        return geometry

    geom_type = geometry.get("type")
    coords = geometry["coordinates"]

    def _signed_area(ring):
        if not ring:
            return 0.0
        area = 0.0
        n = len(ring)
        base_lon = ring[0][0]
        normalized_ring = []
        for p in ring:
            lon = p[0]
            while lon - base_lon > 180:
                lon -= 360
            while lon - base_lon < -180:
                lon += 360
            normalized_ring.append((lon, p[1]))

        for i in range(n):
            p1 = normalized_ring[i]
            p2 = normalized_ring[(i + 1) % n]
            area += (p2[0] - p1[0]) * (p2[1] + p1[1])
        return area / 2.0

    def fix_ring(ring, is_exterior):
        if len(ring) < 4:
            return ring
        area = _signed_area(ring)
        # area > 0 означает против часовой стрелки (CCW)
        # area < 0 означает по часовой стрелке (CW)

        # Для внешних колец нужно CCW (area > 0)
        if is_exterior and area < 0:
            return list(reversed(ring))
        # Для внутренних колец (дырок) нужно CW (area < 0)
        if not is_exterior and area > 0:
            return list(reversed(ring))
        return ring

    if geom_type == "Polygon":
        new_coords = []
        for i, ring in enumerate(coords):
            new_coords.append(fix_ring(ring, is_exterior=(i == 0)))
        geometry["coordinates"] = new_coords
    elif geom_type == "MultiPolygon":
        new_coords = []
        for poly in coords:
            new_poly = []
            for i, ring in enumerate(poly):
                new_poly.append(fix_ring(ring, is_exterior=(i == 0)))
            new_coords.append(new_poly)
        geometry["coordinates"] = new_coords

    return geometry


def _simplify_geometry(geometry, tolerance=0.01):
    """Упрощает геометрию, уменьшая количество точек.
    tolerance контролирует степень упрощения (в градусах).
    0.01 ≈ 1 км на экваторе, 0.05 ≈ 5 км.
    """
    if not isinstance(geometry, dict) or "coordinates" not in geometry:
        return geometry

    try:
        from shapely.geometry import shape, mapping

        polygon = shape(geometry)
        if polygon.is_empty:
            return geometry

        # preserve_topology=True сохраняет правильную топологию
        simplified = polygon.simplify(tolerance, preserve_topology=True)

        if simplified.is_empty:
            return geometry

        # Если упрощение превратило MultiPolygon в Polygon или наоборот
        if simplified.geom_type not in {"Polygon", "MultiPolygon"}:
            return geometry

        result = mapping(simplified)
        return {"type": result["type"], "coordinates": result["coordinates"]}

    except (ImportError, ValueError, TypeError):
        return geometry


def parse_geometry(value):
    """Понимает GeoJSON/WKT и списки OSM way-сегментов из indicators.csv."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
            if text.upper().startswith(("POLYGON", "MULTIPOLYGON")):
                try:
                    from shapely import wkt
                    parsed = wkt.loads(text).__geo_interface__
                except (ImportError, ValueError, TypeError):
                    return None

    geometry = None
    if isinstance(parsed, dict):
        if parsed.get("type") == "Feature":
            geometry = parsed.get("geometry")
        elif parsed.get("type") in {"Polygon", "MultiPolygon"}:
            geometry = parsed
    elif isinstance(parsed, list):
        outer_ways = [
            item for item in parsed
            if isinstance(item, dict)
               and item.get("type") == "way"
               and item.get("role", "outer") == "outer"
               and item.get("geometry")
        ]
        rings = _join_outer_rings(outer_ways)
        if rings:
            geometry = (
                {"type": "Polygon", "coordinates": [rings[0]]}
                if len(rings) == 1
                else {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}
            )

    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        return None
    if not geometry.get("coordinates"):
        return None

    # Validate and repair polygon topology when Shapely is available. Invalid
    # rings can otherwise cause Plotly to fill a large rectangular-looking area.
    try:
        from shapely.geometry import shape, mapping
        polygon = shape(geometry)
        if not polygon.is_valid:
            try:
                from shapely.validation import make_valid
                polygon = make_valid(polygon)
            except (ImportError, AttributeError, ValueError):
                polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None
        if polygon.geom_type == "Polygon":
            repaired = mapping(polygon)
        elif polygon.geom_type == "MultiPolygon":
            repaired = mapping(polygon)
        elif polygon.geom_type == "GeometryCollection":
            parts = [part for part in polygon.geoms if part.geom_type == "Polygon" and not part.is_empty]
            if not parts:
                return None
            from shapely.geometry import MultiPolygon
            repaired = mapping(MultiPolygon(parts))
        else:
            return None
        geometry = {"type": repaired["type"], "coordinates": repaired["coordinates"]}
    except ImportError:
        pass
    except (ValueError, TypeError, KeyError):
        return None

    # Исправляем порядок обхода колец (winding order)
    geometry = _fix_winding_order(geometry)

    geometry = _simplify_geometry(geometry, tolerance=0.01)

    return geometry


def geojson_from_indicators(indicators):
    if "geometry" not in indicators.columns:
        return None, 0
    features = []
    for _, row in indicators.iterrows():
        geometry = parse_geometry(row.get("geometry"))
        if geometry is None:
            continue
        region = str(row.get("region", "")).strip()
        if not region:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "region": region,
                "_region_key": normalize_name(region),
            },
            "geometry": geometry,
        })
    if not features:
        return None, 0
    return {"type": "FeatureCollection", "features": features}, len(features)


def available_indicator_years(indicators, indicator):
    years = []
    if indicator in {"population_change", "population_growth"}:
        population_years = {
            year for year, _ in population_year_columns(indicators)
        }
        return sorted(
            year for year in population_years
            if year - 1 in population_years
        )

    for column in indicators.columns:
        match = re.search(rf"^{re.escape(indicator)}_(\d{{4}})(?:\.0)?$", str(column))
        if match and pd.to_numeric(
                indicators[column], errors="coerce"
        ).notna().any():
            years.append(int(match.group(1)))

    if indicator == "population":
        return [year for year, _ in population_year_columns(indicators)]

    return sorted(set(years))


def build_map_data(indicators, indicator, year):
    region_col = "region"
    if region_col not in indicators.columns:
        raise ValueError("В indicators.csv не найден столбец region.")

    data = indicators[[region_col]].copy()
    data["value"] = indicator_values(indicators, indicator, year)
    data["region_key"] = data[region_col].map(normalize_name)
    data = data.dropna(subset=["value"])
    return data.drop_duplicates("region_key", keep="first")


def make_forecast_long(forecast_df):
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame(columns=["region", "year", "model", "forecast"])

    df = forecast_df.copy()
    if {"region", "year", "population_forecast"}.issubset(df.columns):
        if "model" in df.columns:
            return df.rename(
                columns={"population_forecast": "forecast"}
            )[["region", "year", "model", "forecast"]]
        model_columns = [
            column for column in df.columns
            if column not in {
                "region", "federal_district", "year",
                "lower_95", "upper_95",
            }
               and pd.api.types.is_numeric_dtype(df[column])
        ]
        if model_columns:
            return df.melt(
                id_vars=["region", "year"],
                value_vars=model_columns,
                var_name="model",
                value_name="forecast",
            ).dropna(subset=["forecast"])
        return df.rename(
            columns={"population_forecast": "forecast"}
        ).assign(model="Прогноз модели")[
            ["region", "year", "model", "forecast"]
        ]

    id_columns = [
        column for column in ["region", "federal_district", "year"]
        if column in df.columns
    ]
    if {"region", "year"}.issubset(df.columns):
        model_columns = [
            column for column in df.columns
            if column not in {
                "region", "federal_district", "year",
                "lower_95", "upper_95",
            }
               and pd.api.types.is_numeric_dtype(df[column])
        ]
        if model_columns:
            return df.melt(
                id_vars=id_columns,
                value_vars=model_columns,
                var_name="model",
                value_name="forecast",
            ).dropna(subset=["forecast"])

    return pd.DataFrame(columns=["region", "year", "model", "forecast"])


def forecast_history(indicators, region):
    years = population_year_columns(indicators)
    if not years:
        return pd.DataFrame(columns=["year", "population"])
    region_data = indicators.loc[indicators["region"].map(normalize_name) == normalize_name(region)]
    if region_data.empty:
        return pd.DataFrame(columns=["year", "population"])
    row = region_data.iloc[0]
    return pd.DataFrame([
        {"year": year, "population": pd.to_numeric(row[column], errors="coerce")}
        for year, column in years
    ]).dropna(subset=["population"]).sort_values("year")


def make_comparison(indicators, region_a, region_b, year):
    rows = []
    for label, indicator in INDICATORS.items():
        values_a = indicator_values(indicators.loc[indicators["region"] == region_a], indicator, year)
        values_b = indicator_values(indicators.loc[indicators["region"] == region_b], indicator, year)
        value_a = values_a.iloc[0] if len(values_a) else np.nan
        value_b = values_b.iloc[0] if len(values_b) else np.nan
        rows.append({
            "Показатель": label,
            region_a: value_a,
            region_b: value_b,
        })
    return pd.DataFrame(rows)


INTEGER_INDICATORS = {
    "population", "population_change", "births", "deaths", "natural_change",
}


def is_integer_indicator(indicator):
    return indicator in INTEGER_INDICATORS


def clean_missing_values(frame):
    return frame.replace({"None": np.nan, "none": np.nan, "NULL": np.nan,
                          "null": np.nan, "": np.nan})


def display_model_name(value):
    raw = str(value).strip()
    key = re.sub(r"[^a-zа-я0-9]", "", raw.lower().replace("ё", "е"))
    model_names_ru = {
        "naive": "Наивный метод",
        "naiveforecast": "Наивный метод",
        "naivebaseline": "Наивный базовый метод",
        "baseline": "Базовый метод",
        "randomforest": "Случайный лес",
        "rf": "Случайный лес",
        "gradientboosting": "Градиентный бустинг",
        "gb": "Градиентный бустинг",
        "linearregression": "Линейная регрессия",
        "arima": "ARIMA",
        "exponentialsmoothing": "Экспоненциальное сглаживание",
    }
    if key in model_names_ru:
        return model_names_ru[key]
    normalized = raw.lower().replace("_", " ").replace("-", " ").strip()
    return normalized[:1].upper() + normalized[1:] if normalized else "Метод прогнозирования"


def main():
    st.set_page_config(page_title="Демографическая аналитика регионов РФ", page_icon="📊", layout="wide")
    st.title("Демографическая аналитика регионов России")
    st.caption("Единая панель: карта, прогноз и сопоставление двух субъектов РФ.")

    indicators = read_csv(INDICATORS_FILE)
    if indicators is None:
        st.error(f"Не найден файл с показателями: {INDICATORS_FILE}")
        st.stop()
    if "region" not in indicators.columns:
        st.error("В indicators.csv отсутствует обязательный столбец region.")
        st.stop()

    indicators["region"] = indicators["region"].astype(str).str.strip()
    indicators = indicators[indicators["region"].ne("")].drop_duplicates("region", keep="first")
    regions = sorted(indicators["region"].dropna().unique().tolist())
    years_available = [year for year, _ in population_year_columns(indicators)]
    if not regions or not years_available:
        st.error("Не удалось определить регионы или годы численности населения.")
        st.stop()

    map_col, forecast_col, compare_col = st.columns([1.05, 1.1, 1.0], gap="medium")
    with map_col:
        st.header("1. Карта демографического показателя")
        map_controls = st.columns([2, 1])
        with map_controls[0]:
            selected_indicator_label = st.selectbox("Показатель карты", list(INDICATORS.keys()), key="map_indicator")
        indicator = INDICATORS[selected_indicator_label]
        map_years = available_indicator_years(indicators, indicator)
        with map_controls[1]:
            selected_year = st.selectbox("Год", map_years or years_available,
                                         index=len(map_years or years_available) - 1, key="map_year")

        map_data = build_map_data(indicators, indicator, selected_year)
        geojson, geometry_count = geojson_from_indicators(indicators)
        if geojson is None:
            st.warning("В столбце `geometry` файла `indicators.csv` не найдены распознаваемые полигоны субъектов РФ.")
        else:
            geometry_keys = {
                feature["properties"]["_region_key"]
                for feature in geojson["features"]
            }
            matched = map_data[map_data["region_key"].isin(geometry_keys)].copy()
            matched_keys = set(matched["region_key"])
            matched_geojson = {
                "type": "FeatureCollection",
                "features": [
                    feature for feature in geojson["features"]
                    if feature["properties"]["_region_key"] in matched_keys
                ],
            }
            if not matched.empty:
                fig = px.choropleth(
                    matched,
                    geojson=matched_geojson,
                    locations="region_key",
                    featureidkey="properties._region_key",
                    color="value",
                    hover_name="region",
                    hover_data={"region_key": False, "value": ":,.0f" if is_integer_indicator(indicator) else ":,.2f"},
                    projection="mercator",
                    labels={"value": selected_indicator_label},
                )
                fig.update_geos(
                    visible=False,
                    #lonaxis=dict(range=[15, 195]),  # Западная и восточная границы РФ
                    #lataxis=dict(range=[55, 85]),  # Южная и северная границы РФ
                    fitbounds="locations",
                    projection_type="transverse mercator",
                    bgcolor="#0e111700"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),  # Увеличил нижний отступ для легенды
                    coloraxis_colorbar=dict(
                        title=selected_indicator_label,
                        orientation="h",  # Горизонтальная ориентация
                        x=0.5,  # Центрирование по горизонтали
                        xanchor="center",
                        y=0.2,  # Позиция под картой
                        yanchor="top",
                        len=1.0,  # Длина цветовой шкалы
                        thickness=15,  # Толщина
                        bgcolor="rgba(0,0,0,0)",  # Прозрачный фон
                        bordercolor="rgba(0,0,0,0)",  # Прозрачная граница
                        borderwidth=0,
                        outlinewidth=0,
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",  # Прозрачный фон страницы
                    plot_bgcolor="rgba(0,0,0,0)",  # Прозрачный фон графика
                    #height=620,
                )
                st.plotly_chart(fig, use_container_width=True)

    with forecast_col:
        st.header("2. Историческая численность населения и прогноз")
        forecast_df = read_csv(FORECAST_FILE)
        forecast_long = make_forecast_long(forecast_df)
        if not forecast_long.empty:
            forecast_long["year"] = pd.to_numeric(forecast_long["year"], errors="coerce")
            forecast_long["forecast"] = pd.to_numeric(forecast_long["forecast"], errors="coerce")
            forecast_long["region_key"] = forecast_long["region"].map(normalize_name)
            forecast_long = forecast_long[
                forecast_long["year"].notna() & forecast_long["forecast"].notna()
                ].copy()

        forecast_regions = sorted(
            forecast_long["region"].astype(str).unique().tolist()) if not forecast_long.empty else []
        if not forecast_regions:
            st.warning("В `population_forecast.csv` нет распознаваемых прогнозных строк.")
        else:
            selected_region = st.selectbox(
                "Субъект РФ для прогноза",
                forecast_regions,
                key="forecast_region",
            )
            history = forecast_history(indicators, selected_region)
            selected_key = normalize_name(selected_region)
            region_forecasts = forecast_long[
                forecast_long["region_key"] == selected_key
                ].copy()

            if history.empty:
                st.warning("Для выбранного субъекта нет исторических данных населения в indicators.csv.")
            elif region_forecasts.empty:
                st.warning("В файле прогнозов нет строк для выбранного субъекта.")
            else:
                last_year = int(history["year"].max())
                future_models = region_forecasts[
                    region_forecasts["year"] > last_year
                    ].copy()
                chart = go.Figure()
                chart.add_trace(go.Scatter(
                    x=history["year"],
                    y=history["population"],
                    mode="lines",
                    name="Фактическая численность населения",
                    line=dict(width=3),
                    hovertemplate="Год: %{x:.0f}<br>Численность: %{y:,.0f}<extra></extra>",
                ))

                last_value_rows = history.loc[history["year"] == last_year, "population"]
                if not last_value_rows.empty:
                    anchor_value = float(last_value_rows.iloc[-1])
                    model_groups = []
                    for model_name, group in future_models.groupby("model", sort=False):
                        group = group.sort_values("year")
                        if group.empty:
                            continue
                        model_groups.append((model_name, group, float(group.iloc[-1]["forecast"])))
                    model_groups.sort(key=lambda item: item[2], reverse=True)
                    for model_name, group, _last_forecast in model_groups:
                        plot_x = [last_year] + group["year"].astype(int).tolist()
                        plot_y = [anchor_value] + group["forecast"].astype(float).tolist()
                        chart.add_trace(go.Scatter(
                            x=plot_x,
                            y=plot_y,
                            mode="lines+markers",
                            name=display_model_name(model_name),
                            line=dict(dash="dash"),
                            hovertemplate="Год: %{x:.0f}<br>Прогноз: %{y:,.0f}<extra>%{fullData.name}</extra>",
                        ))

                chart.update_layout(
                    xaxis_title="Год",
                    yaxis_title="Численность населения",
                    hovermode="x unified",
                    yaxis=dict(rangemode="tozero"),
                    legend=dict(
                        title="Метод прогнозирования",
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.28,
                        yanchor="top",
                    ),
                    margin=dict(l=15, r=15, t=20, b=100),
                    height=500,
                )
                st.plotly_chart(chart, use_container_width=True)

    with compare_col:
        st.header("3. Сравнение двух субъектов")
        compare_controls = st.columns([2, 2, 1])
        with compare_controls[0]:
            region_a = st.selectbox("Субъект слева", regions, index=0, key="compare_region_a")
        with compare_controls[1]:
            other_regions = [region for region in regions if region != region_a]
            region_b = st.selectbox("Субъект справа", other_regions or regions, index=0, key="compare_region_b")
        with compare_controls[2]:
            comparison_year = st.selectbox("Год сравнения", years_available, index=len(years_available) - 1,
                                           key="compare_year")

        comparison = make_comparison(indicators, region_a, region_b, comparison_year)
        comparison = clean_missing_values(comparison)
        comparison = comparison[comparison["Показатель"].fillna("").astype(str).str.strip().ne("")]

        def format_comparison_value(value, indicator_label):
            if pd.isna(value):
                return "Нет данных"
            indicator = INDICATORS.get(indicator_label)
            if is_integer_indicator(indicator):
                return f"{value:,.0f}".replace(",", " ")
            return f"{value:,.2f}".replace(",", " ")

        for column in (region_a, region_b):
            comparison[column] = [
                format_comparison_value(value, label)
                for label, value in zip(comparison["Показатель"], comparison[column])
            ]
        comparison = comparison.replace({"None": "Нет данных", "nan": "Нет данных"})
        styled_comparison = comparison.style.map(
            lambda value: "color: #9CA3AF" if value == "Нет данных" else ""
        )
        st.dataframe(
            styled_comparison,
            use_container_width=True,
            hide_index=True,
            height=390,
        )


if __name__ == "__main__":
    main()