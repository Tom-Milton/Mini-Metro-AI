from pydantic import BaseModel, field_validator

from src import types


class Station(BaseModel):
    position: types.Position
    shape: types.StationShape

    @field_validator("shape", mode="before")
    @classmethod
    def parse_shape(cls, v: str | types.StationShape) -> types.StationShape:
        if isinstance(v, str):
            return types.StationShape.from_label(v)
        return v


class GameState(BaseModel):
    stations: list[Station]

    def sorted(self) -> "GameState":
        return GameState(stations=sorted(self.stations, key=lambda s: s.position))
