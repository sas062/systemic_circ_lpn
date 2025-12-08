from dataclasses import dataclass

@dataclass
class Vessel:
    name: str
    start_node: str
    end_node: str
    R: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "R": self.R,
        }
