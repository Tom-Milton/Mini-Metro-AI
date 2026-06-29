from pydantic import BaseModel, field_validator

from src.types import Position, StationShape


class Station(BaseModel):
    position: Position
    shape: StationShape

    @field_validator("shape", mode="before")
    @classmethod
    def parse_shape(cls, v: str | StationShape) -> StationShape:
        if isinstance(v, str):
            return StationShape.from_label(v)
        return v


class GameState(BaseModel):
    stations: list[Station]

    def sorted(self) -> "GameState":
        return GameState(stations=sorted(self.stations, key=lambda s: s.position))
