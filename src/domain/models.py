from pydantic import BaseModel, field_validator, computed_field

from src.domain import types


class Station(BaseModel):
    position: types.Position
    shape: types.StationShape

    @field_validator("shape", mode="before")
    @classmethod
    def parse_shape(cls, v: str | types.StationShape) -> types.StationShape:
        if isinstance(v, str):
            return types.StationShape.from_label(v)
        return v

    @computed_field
    @property
    def id(self) -> types.StationId:
        return f"{self.shape.value.name}-{self.position[0]}-{self.position[1]}"


class Line(BaseModel):
    id: types.LineColour
    station_ids: list[types.StationId]


class Menu(BaseModel):
    line_id: types.LineColour
    position: types.Position
    state: types.MenuState


class GameState(BaseModel):
    stations: dict[types.StationId, Station]
    lines: dict[types.LineColour, Line]
    menus: dict[types.LineColour, Menu]
