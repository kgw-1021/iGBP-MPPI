from dataclasses import dataclass

@dataclass
class State:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

@dataclass
class Gaussian:
    xx: float
    yy: float
    xy: float = 0.0
    vxvx: float = 100.0
    vyvy: float = 100.0
    vxvy: float = 0.0

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    is_wall: bool = False