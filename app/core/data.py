"""Visa demand data for 21 country blocs x 5 EB categories.

Demand n_g is the employment-based green-card BACKLOG (pending demand) per
(country, category) group, calibrated to the documented official structure of the
US employment-based queue: a total backlog of about 1.8 million, with India ~63%
and China ~14% (together ~77%), the rest a long tail led by the Philippines and
Mexico (per Cato Institute analysis of USCIS/Department of State data, 2024; and
the Department of State Annual Report of Immigrant Visa Applicants). Priority
dates d_g give the elapsed wait w_g = T_actual - d_g (India EB-2/EB-3 ~13 years,
China EB-5 ~10 years, most other blocs near-current). This is a representative
instance consistent with the official structure, not an official per-applicant
census; conclusions are shown robust to +/-20% demand perturbation.
"""

from app.core.config import COUNTRIES, CATEGORIES, T_ACTUAL, K_BASE, P_C, V

VISA_DATA: dict[str, dict[str, dict[str, int]]] = {
    # India ~63% of the EB backlog, overwhelmingly EB-2 / EB-3 (decade-plus waits).
    "India": {
        "EB-1": {"n": 60_000, "d": 2023},
        "EB-2": {"n": 690_000, "d": 2013},
        "EB-3": {"n": 350_000, "d": 2013},
        "EB-4": {"n": 16_000, "d": 2022},
        "EB-5": {"n": 18_000, "d": 2020},
    },
    # China ~14%, with a large EB-5 backlog.
    "China": {
        "EB-1": {"n": 30_000, "d": 2023},
        "EB-2": {"n": 95_000, "d": 2021},
        "EB-3": {"n": 38_000, "d": 2021},
        "EB-4": {"n": 2_000, "d": 2022},
        "EB-5": {"n": 87_000, "d": 2016},
    },
    "Filipinas": {
        "EB-1": {"n": 1_000, "d": 2026},
        "EB-2": {"n": 3_000, "d": 2025},
        "EB-3": {"n": 38_000, "d": 2022},
        "EB-4": {"n": 2_500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Mexico": {
        "EB-1": {"n": 1_500, "d": 2026},
        "EB-2": {"n": 4_000, "d": 2025},
        "EB-3": {"n": 10_000, "d": 2023},
        "EB-4": {"n": 24_000, "d": 2021},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Afganistan": {
        "EB-1": {"n": 100, "d": 2026},
        "EB-2": {"n": 400, "d": 2025},
        "EB-3": {"n": 1_200, "d": 2023},
        "EB-4": {"n": 600, "d": 2022},
        "EB-5": {"n": 200, "d": 2026},
    },
    "Irak": {
        "EB-1": {"n": 100, "d": 2026},
        "EB-2": {"n": 400, "d": 2025},
        "EB-3": {"n": 1_000, "d": 2023},
        "EB-4": {"n": 400, "d": 2022},
        "EB-5": {"n": 100, "d": 2026},
    },
    "Corea del Sur": {
        "EB-1": {"n": 2_000, "d": 2026},
        "EB-2": {"n": 5_000, "d": 2025},
        "EB-3": {"n": 6_500, "d": 2023},
        "EB-4": {"n": 1_000, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Pakistan": {
        "EB-1": {"n": 1_000, "d": 2026},
        "EB-2": {"n": 4_000, "d": 2025},
        "EB-3": {"n": 7_000, "d": 2023},
        "EB-4": {"n": 1_500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Iran": {
        "EB-1": {"n": 1_000, "d": 2026},
        "EB-2": {"n": 4_000, "d": 2025},
        "EB-3": {"n": 5_000, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Taiwan": {
        "EB-1": {"n": 1_500, "d": 2026},
        "EB-2": {"n": 3_000, "d": 2025},
        "EB-3": {"n": 4_000, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 1_000, "d": 2026},
    },
    "Brasil": {
        "EB-1": {"n": 1_000, "d": 2026},
        "EB-2": {"n": 3_000, "d": 2025},
        "EB-3": {"n": 6_500, "d": 2023},
        "EB-4": {"n": 1_000, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Canada": {
        "EB-1": {"n": 2_000, "d": 2026},
        "EB-2": {"n": 3_500, "d": 2025},
        "EB-3": {"n": 3_500, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Reino Unido": {
        "EB-1": {"n": 2_000, "d": 2026},
        "EB-2": {"n": 3_000, "d": 2025},
        "EB-3": {"n": 3_000, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Nigeria": {
        "EB-1": {"n": 800, "d": 2026},
        "EB-2": {"n": 3_000, "d": 2025},
        "EB-3": {"n": 7_000, "d": 2023},
        "EB-4": {"n": 1_700, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Japon": {
        "EB-1": {"n": 1_500, "d": 2026},
        "EB-2": {"n": 2_500, "d": 2025},
        "EB-3": {"n": 3_000, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Bangladesh": {
        "EB-1": {"n": 500, "d": 2026},
        "EB-2": {"n": 3_000, "d": 2025},
        "EB-3": {"n": 7_500, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Colombia": {
        "EB-1": {"n": 800, "d": 2026},
        "EB-2": {"n": 2_500, "d": 2025},
        "EB-3": {"n": 5_700, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Alemania": {
        "EB-1": {"n": 1_500, "d": 2026},
        "EB-2": {"n": 2_500, "d": 2025},
        "EB-3": {"n": 3_000, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Vietnam": {
        "EB-1": {"n": 500, "d": 2026},
        "EB-2": {"n": 1_500, "d": 2025},
        "EB-3": {"n": 8_000, "d": 2023},
        "EB-4": {"n": 2_500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    "Etiopia": {
        "EB-1": {"n": 300, "d": 2026},
        "EB-2": {"n": 1_500, "d": 2025},
        "EB-3": {"n": 4_200, "d": 2023},
        "EB-4": {"n": 500, "d": 2022},
        "EB-5": {"n": 500, "d": 2026},
    },
    # Rest of the World: the long tail of all remaining countries (~10%).
    "Resto del Mundo": {
        "EB-1": {"n": 20_000, "d": 2026},
        "EB-2": {"n": 45_000, "d": 2025},
        "EB-3": {"n": 70_000, "d": 2023},
        "EB-4": {"n": 30_000, "d": 2022},
        "EB-5": {"n": 7_500, "d": 2026},
    },
}

# Data provenance metadata
REAL_DEMAND: dict[str, set[str]] = {
    "India": {"EB-1", "EB-2", "EB-3", "EB-4", "EB-5"},
    "China": {"EB-1", "EB-2", "EB-3", "EB-4", "EB-5"},
    "Filipinas": {"EB-2", "EB-3", "EB-4"},
    "Mexico": {"EB-2", "EB-3", "EB-4"},
}

EST_DEM_CATS: set[str] = {"EB-1", "EB-5"}

DHS_COUNTRIES: set[str] = {
    "Afganistan", "Irak", "Corea del Sur", "Pakistan", "Iran", "Taiwan",
    "Brasil", "Canada", "Reino Unido", "Nigeria", "Japon", "Bangladesh",
    "Colombia", "Alemania", "Vietnam", "Etiopia",
}


def demand_source(country: str, category: str) -> str:
    if country in REAL_DEMAND and category in REAL_DEMAND[country]:
        return "REAL"
    if country in DHS_COUNTRIES and category in EST_DEM_CATS:
        return "EST-DEM"
    if country in DHS_COUNTRIES:
        return "EST"
    if country == "Resto del Mundo" and category in EST_DEM_CATS:
        return "EST-DEM"
    if country == "Resto del Mundo":
        return "EST"
    return "EST-DEM"


def build_groups() -> list[dict]:
    groups: list[dict] = []
    idx = 0
    for country in COUNTRIES:
        for cat in CATEGORIES:
            entry = VISA_DATA[country][cat]
            w = T_ACTUAL - entry["d"]
            groups.append({
                "index": idx,
                "country": country,
                "category": cat,
                "n": entry["n"],
                "d": entry["d"],
                "w": w,
            })
            idx += 1
    return groups


def compute_spillover(groups: list[dict]) -> dict[str, int]:
    demand_by_cat: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for g in groups:
        demand_by_cat[g["category"]] += g["n"]

    d4 = demand_by_cat["EB-4"]
    d5 = demand_by_cat["EB-5"]
    d1 = demand_by_cat["EB-1"]
    d2 = demand_by_cat["EB-2"]

    k4_eff = K_BASE["EB-4"]
    k5_eff = K_BASE["EB-5"]
    s4 = max(0, k4_eff - d4)
    s5 = max(0, k5_eff - d5)

    k1_eff = K_BASE["EB-1"] + s4 + s5
    s1 = max(0, k1_eff - d1)

    k2_eff = K_BASE["EB-2"] + s1
    s2 = max(0, k2_eff - d2)

    k3_eff = K_BASE["EB-3"] + s2

    return {
        "EB-1": k1_eff, "EB-2": k2_eff, "EB-3": k3_eff,
        "EB-4": k4_eff, "EB-5": k5_eff,
    }


def compute_country_caps(groups: list[dict]) -> dict[str, int]:
    country_demand: dict[str, int] = {}
    for g in groups:
        country_demand[g["country"]] = country_demand.get(g["country"], 0) + g["n"]

    caps: dict[str, int] = {}
    sum_others = 0
    for country in COUNTRIES:
        if country != "Resto del Mundo":
            caps[country] = P_C
            sum_others += min(country_demand[country], P_C)

    caps["Resto del Mundo"] = max(P_C, V - sum_others)
    return caps
