import json
from pathlib import Path
from typing import Any


class JsonGateway:
    def load(self, path: Path) -> Any:
        with open(path) as f:
            return json.load(f)

    def save(self, path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f)
