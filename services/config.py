import tomllib, numpy

from dataclasses import dataclass
from pathlib import Path

# Dataclasses
@dataclass
class FileSettings:
    output_dir: str

@dataclass
class TrafficSettings:
    speeds: list[int]
    speed_probs: list[float]
    max_phases: int

@dataclass
class DataGenSettings:
    diurnal: numpy.array

    max_x_accept: float
    pedestrian_cross_prob: float
    heavy_left_prob: float
    burst_traffic_prob: float

    def __post_init__(self):
        self.diurnal = numpy.array(self.diurnal)

@dataclass
class AISettings:
    epochs: int
    batch: int
    lr: float
    hidden: int
    seed: int

@dataclass
class ThemeSettings:
    phase_group_colors: list[str]
    signal_colors: dict[str, str]
    colors: dict[str, str]

# Loader
@dataclass
class Settings:
    file_settings:      FileSettings
    traffic_settings:   TrafficSettings
    data_gen_settings:  DataGenSettings
    ai_settings:        AISettings
    theme_settings:     ThemeSettings

def load(path: str = "settings.toml") -> Settings:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    return Settings(
        file_settings       = FileSettings(**data["file_settings"]),
        traffic_settings    = TrafficSettings(**data["traffic_settings"]),
        data_gen_settings   = DataGenSettings(**data["data_gen_settings"]),
        ai_settings         = AISettings(**data["ai_settings"]),
        theme_settings      = ThemeSettings(**data["theme_settings"])
    )

SETTINGS = load()