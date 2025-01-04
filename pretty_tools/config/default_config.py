import rich
import yaml
from pathlib import Path


class X_Config:

    def load_yaml(self, file_path: Path):
        if file_path.exists():
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        else:
            rich.print(f"[yellow]Config file not found: {file_path}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "r") as f:
                return yaml.dump(f, sort_keys=False)
