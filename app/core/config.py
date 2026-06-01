"""Configuration parameters for the MOHHO Green Card optimization problem."""

V: int = 140_000
V_TOTAL: int = 366_000
P_C: int = 25_620
T_ACTUAL: int = 2026

K_BASE: dict[str, int] = {
    "EB-1": 40_040,
    "EB-2": 40_040,
    "EB-3": 40_040,
    "EB-4": 9_940,
    "EB-5": 9_940,
}

POPULATION_SIZE: int = 50
MAX_ITERATIONS: int = 500
NUM_RUNS: int = 30
ARCHIVE_SIZE: int = 100
SEED_BASE: int = 42
LB: float = 0.0
UB: float = 1.0
BETA_LEVY: float = 1.5

NUM_COUNTRIES: int = 21
NUM_CATEGORIES: int = 5
NUM_GROUPS: int = NUM_COUNTRIES * NUM_CATEGORIES

COUNTRIES: list[str] = [
    "India", "China", "Filipinas", "Mexico",
    "Afganistan", "Irak",
    "Corea del Sur", "Pakistan", "Iran", "Taiwan",
    "Brasil", "Canada", "Reino Unido", "Nigeria",
    "Japon", "Bangladesh", "Colombia", "Alemania",
    "Vietnam", "Etiopia",
    "Resto del Mundo",
]

CATEGORIES: list[str] = ["EB-1", "EB-2", "EB-3", "EB-4", "EB-5"]

CATS_DESC: dict[str, str] = {
    "EB-1": "Extraordinarios",
    "EB-2": "Profesionales",
    "EB-3": "Calificados",
    "EB-4": "Especiales/SIV",
    "EB-5": "Inversores",
}

FLAGS: dict[str, str] = {
    "India": "\U0001f1ee\U0001f1f3", "China": "\U0001f1e8\U0001f1f3",
    "Filipinas": "\U0001f1f5\U0001f1ed", "Mexico": "\U0001f1f2\U0001f1fd",
    "Afganistan": "\U0001f1e6\U0001f1eb", "Irak": "\U0001f1ee\U0001f1f6",
    "Corea del Sur": "\U0001f1f0\U0001f1f7", "Pakistan": "\U0001f1f5\U0001f1f0",
    "Iran": "\U0001f1ee\U0001f1f7", "Taiwan": "\U0001f1f9\U0001f1fc",
    "Brasil": "\U0001f1e7\U0001f1f7", "Canada": "\U0001f1e8\U0001f1e6",
    "Reino Unido": "\U0001f1ec\U0001f1e7", "Nigeria": "\U0001f1f3\U0001f1ec",
    "Japon": "\U0001f1ef\U0001f1f5", "Bangladesh": "\U0001f1e7\U0001f1e9",
    "Colombia": "\U0001f1e8\U0001f1f4", "Alemania": "\U0001f1e9\U0001f1ea",
    "Vietnam": "\U0001f1fb\U0001f1f3", "Etiopia": "\U0001f1ea\U0001f1f9",
    "Resto del Mundo": "\U0001f30d",
}
