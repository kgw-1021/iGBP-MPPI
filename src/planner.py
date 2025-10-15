import numpy as np
import matplotlib.pyplot as plt
from core.types import State, Obstacle
from core.robot import Robot
from planning.gbp import perform_gbp_inference
from planning.mppi import perform_mppi
from visualization.plot import visualize
from utils.utils import distance 
from typing import List, Tuple
import copy

class GBP_MPPI_Planner:
    def __init__(self, num_robots=3, scenario='narrow_passage'):   # available : 'narrow_passage', 'obstacle'
        # Parameters
        self.num_robots = num_robots
        self.horizon_steps = 15
        self.comm_range = 200
        self.safety_dist = 20
        self.gbp_iterations = 3
        self.mppi_samples = 50
        self.iterative_cycles = 2
        self.dt = 0.2
        self.lambda_mppi = 1.0
        self.max_accel = 15
        self.max_vel = 30
        
        # Scenario
        self.scenario = scenario
        self.obstacles = self._create_obstacles()
        self.robots = self._create_robots()
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.time_step = 0
        
    def _create_obstacles(self) -> List[Obstacle]:
        if self.scenario == 'narrow_passage':
            obstacles = []
            gap_size = 120
            wall_x = 400
            wall_thickness = 20
            
            # Top wall
            for y in range(50, int(300 - gap_size/2), 30):
                obstacles.append(Obstacle(wall_x, y, wall_thickness, is_wall=True))
            
            # Bottom wall
            for y in range(int(300 + gap_size/2), 550, 30):
                obstacles.append(Obstacle(wall_x, y, wall_thickness, is_wall=True))
            
            return obstacles
        elif self.scenario == 'obstacle':
            return [Obstacle(400, 300, 40, is_wall=False)]
        else:  # open
            return []
    
    def _create_robots(self) -> List[Robot]:
        robots = []
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
        
        for i in range(self.num_robots):
            if self.scenario == 'narrow_passage':
                if i < self.num_robots / 2:
                    # Left side going right
                    start = State(150, 250 + i * 50)
                    goal = State(650, 350 - i * 50)
                else:
                    # Right side going left
                    idx = i - int(self.num_robots / 2)
                    start = State(650, 250 + idx * 50)
                    goal = State(150, 350 - idx * 50)
            else:
                # Circular arrangement
                angle = (i / self.num_robots) * 2 * np.pi
                radius = 180
                start = State(400 + radius * np.cos(angle), 
                            300 + radius * np.sin(angle))
                goal = State(400 + radius * np.cos(angle + np.pi),
                           300 + radius * np.sin(angle + np.pi))
            
            robots.append(Robot(i, start, goal, colors[i % len(colors)]))
        
        return robots
    
    def find_neighbors(self):
        """Update neighbor lists based on communication range"""
        for robot in self.robots:
            robot.neighbors = []
            for other in self.robots:
                if other.id != robot.id:
                    if self.distance(robot.state, other.state) <= self.comm_range:
                        robot.neighbors.append(other.id)
    
    
    def step(self):
        """Single planning and execution step"""
        # Find neighbors
        self.find_neighbors()
        
        # Iterative GBP ↔ MPPI refinement
        for cycle in range(self.iterative_cycles):
            # GBP inference
            for robot in self.robots:
                mppi_prior_mean = robot.mppi_mean if cycle > 0 else None
                mppi_prior_sigma = robot.mppi_sigma if cycle > 0 else None
                
                mu, sigma, uncertainty = self.perform_gbp_inference(
                    robot, mppi_prior_mean, mppi_prior_sigma
                )
                robot.gbp_mu = mu
                robot.gbp_sigma = sigma
                robot.gbp_uncertainty = uncertainty
            
            # MPPI sampling
            for robot in self.robots:
                rollouts, weights, mean, sigma, uncertainty = self.perform_mppi(robot)
                robot.mppi_rollouts = rollouts
                robot.mppi_weights = weights
                robot.mppi_mean = mean
                robot.mppi_sigma = sigma
                robot.mppi_uncertainty = uncertainty
        
        # Select best trajectory
        for robot in self.robots:
            if robot.mppi_weights:
                best_idx = np.argmax(robot.mppi_weights)
                robot.selected_trajectory = robot.mppi_rollouts[best_idx]
            else:
                robot.selected_trajectory = robot.mppi_mean
        
        # Execute first step
        for robot in self.robots:
            if robot.selected_trajectory and len(robot.selected_trajectory) > 1:
                robot.trajectory_history.append((robot.state.x, robot.state.y))
                if len(robot.trajectory_history) > 100:
                    robot.trajectory_history.pop(0)
                
                next_state = robot.selected_trajectory[1]
                robot.state = copy.deepcopy(next_state)
        
        self.time_step += 1