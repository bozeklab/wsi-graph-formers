from dataclasses import dataclass

@dataclass
class GraphConfig:
    json_path: str
    output_path: str
    radius: float = 50.0
    max_nodes: int = 1000
    figsize: tuple = (10, 10)
