from core.types import State
import numpy as np


def distance(self, p1: State, p2: State) -> float:
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)