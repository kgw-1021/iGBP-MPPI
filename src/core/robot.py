from typing import List, Tuple
from dataclasses import dataclass
import copy
from .types import State, Gaussian

class Robot:
    def __init__(self, robot_id: int, start: State, goal: State, color: str):
        self.id = robot_id
        self.state = copy.deepcopy(start)
        self.goal = goal
        self.color = color
        
        # GBP results
        self.gbp_mu: List[State] = []
        self.gbp_sigma: List[Gaussian] = []
        self.gbp_uncertainty: List[float] = []
        
        # MPPI results
        self.mppi_rollouts: List[List[State]] = []
        self.mppi_weights: List[float] = []
        self.mppi_mean: List[State] = []
        self.mppi_sigma: List[Gaussian] = []
        self.mppi_uncertainty: List[float] = []
        
        # Selected trajectory
        self.selected_trajectory: List[State] = []
        
        # History
        self.trajectory_history: List[Tuple[float, float]] = []
        
        # Neighbors
        self.neighbors: List[int] = []