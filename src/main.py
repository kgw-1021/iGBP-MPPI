"""
Iterative GBP ↔ MPPI Framework for Multi-Robot Trajectory Planning
Narrow Passage Scenario
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy

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

class GBPMPPIPlanner:
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
    
    def distance(self, p1: State, p2: State) -> float:
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def find_neighbors(self):
        """Update neighbor lists based on communication range"""
        for robot in self.robots:
            robot.neighbors = []
            for other in self.robots:
                if other.id != robot.id:
                    if self.distance(robot.state, other.state) <= self.comm_range:
                        robot.neighbors.append(other.id)
    
    def perform_gbp_inference(self, robot: Robot, mppi_mean_prior=None, 
                             mppi_sigma_prior=None) -> Tuple[List[State], List[Gaussian]]:
        """GBP inference with optional MPPI posterior feedback"""
        T = self.horizon_steps
        
        # Initialize
        if mppi_mean_prior and len(mppi_mean_prior) > 0:
            mu = copy.deepcopy(mppi_mean_prior)
            sigma = copy.deepcopy(mppi_sigma_prior)
        else:
            # Cold start: straight line to goal
            mu = []
            sigma = []
            for t in range(T + 1):
                alpha = t / T
                mu.append(State(
                    robot.state.x + alpha * (robot.goal.x - robot.state.x),
                    robot.state.y + alpha * (robot.goal.y - robot.state.y),
                    (robot.goal.x - robot.state.x) / (T * self.dt),
                    (robot.goal.y - robot.state.y) / (T * self.dt)
                ))
                sigma.append(Gaussian(1000, 1000))
        
        # Message passing
        for iteration in range(self.gbp_iterations):
            new_mu = copy.deepcopy(mu)
            new_sigma = copy.deepcopy(sigma)
            
            for t in range(T + 1):
                # Dynamics factor
                if t < T:
                    predicted_x = mu[t].x + mu[t].vx * self.dt
                    predicted_y = mu[t].y + mu[t].vy * self.dt
                    new_mu[t].x += (predicted_x - mu[t+1].x) * 0.05
                    new_mu[t].y += (predicted_y - mu[t+1].y) * 0.05
                    new_sigma[t].xx *= 0.95
                    new_sigma[t].yy *= 0.95
                
                # Goal factor
                if t == T:
                    dx = robot.goal.x - mu[t].x
                    dy = robot.goal.y - mu[t].y
                    dist = np.sqrt(dx**2 + dy**2)
                    strength = min(1.0, dist / 100)
                    new_mu[t].x += dx * 0.5 * strength
                    new_mu[t].y += dy * 0.5 * strength
                    new_sigma[t].xx = min(new_sigma[t].xx, 50)
                    new_sigma[t].yy = min(new_sigma[t].yy, 50)
                
                # Obstacle factors
                for obs in self.obstacles:
                    dx = mu[t].x - obs.x
                    dy = mu[t].y - obs.y
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = obs.radius + self.safety_dist
                    
                    if dist < min_dist:
                        factor = (min_dist - dist) / min_dist
                        new_mu[t].x += (dx / dist) * factor * 20
                        new_mu[t].y += (dy / dist) * factor * 20
                        new_sigma[t].xx += 100
                        new_sigma[t].yy += 100
                
                # Inter-robot collision factors
                for neighbor_id in robot.neighbors:
                    neighbor = self.robots[neighbor_id]
                    if len(neighbor.gbp_mu) > t:
                        dx = mu[t].x - neighbor.gbp_mu[t].x
                        dy = mu[t].y - neighbor.gbp_mu[t].y
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        if dist < self.safety_dist * 2:
                            factor = (self.safety_dist * 2 - dist) / (self.safety_dist * 2)
                            new_mu[t].x += (dx / dist) * factor * 15
                            new_mu[t].y += (dy / dist) * factor * 15
                            new_sigma[t].xx += 150
                            new_sigma[t].yy += 150
                
                # Smoothness
                if 0 < t < T:
                    acc_x = mu[t+1].x - 2 * mu[t].x + mu[t-1].x
                    acc_y = mu[t+1].y - 2 * mu[t].y + mu[t-1].y
                    new_mu[t].x += -acc_x * 0.1
                    new_mu[t].y += -acc_y * 0.1
                
                # MPPI posterior feedback (precision fusion)
                if mppi_sigma_prior and len(mppi_sigma_prior) > t:
                    precision_gbp = 1.0 / new_sigma[t].xx
                    precision_mppi = 1.0 / mppi_sigma_prior[t].xx
                    fused_precision = precision_gbp + precision_mppi * 0.5
                    new_sigma[t].xx = 1.0 / fused_precision
                    
                    precision_gbp_y = 1.0 / new_sigma[t].yy
                    precision_mppi_y = 1.0 / mppi_sigma_prior[t].yy
                    fused_precision_y = precision_gbp_y + precision_mppi_y * 0.5
                    new_sigma[t].yy = 1.0 / fused_precision_y
            
            mu = new_mu
            sigma = new_sigma
        
        # Compute uncertainty
        uncertainty = [np.sqrt(sig.xx + sig.yy) for sig in sigma]
        
        return mu, sigma, uncertainty
    
    def perform_mppi(self, robot: Robot):
        """MPPI sampling from GBP belief"""
        if not robot.gbp_mu or len(robot.gbp_mu) == 0:
            return [], [], [], [], []
        
        rollouts = []
        costs = []
        
        # Generate samples
        for s in range(self.mppi_samples):
            rollout = [copy.deepcopy(robot.state)]
            
            for t in range(1, len(robot.gbp_mu)):
                mu = robot.gbp_mu[t]
                sigma = robot.gbp_sigma[t]
                prev_state = rollout[t - 1]
                
                # Adaptive noise
                noise_scale = min(1.0, np.sqrt(sigma.xx) / 60)
                ax_noise = (np.random.rand() - 0.5) * 2 * self.max_accel * noise_scale * 0.4
                ay_noise = (np.random.rand() - 0.5) * 2 * self.max_accel * noise_scale * 0.4
                
                # Desired acceleration towards mean
                desired_vx = (mu.x - prev_state.x) / self.dt
                desired_vy = (mu.y - prev_state.y) / self.dt
                ax_mean = (desired_vx - prev_state.vx) / self.dt
                ay_mean = (desired_vy - prev_state.vy) / self.dt
                
                # Add noise
                ax = ax_mean + ax_noise
                ay = ay_mean + ay_noise
                
                # Clamp acceleration
                accel_mag = np.sqrt(ax**2 + ay**2)
                if accel_mag > self.max_accel:
                    ax = (ax / accel_mag) * self.max_accel
                    ay = (ay / accel_mag) * self.max_accel
                
                # Integrate velocity
                vx = prev_state.vx + ax * self.dt
                vy = prev_state.vy + ay * self.dt
                
                # Clamp velocity
                vel_mag = np.sqrt(vx**2 + vy**2)
                if vel_mag > self.max_vel:
                    vx = (vx / vel_mag) * self.max_vel
                    vy = (vy / vel_mag) * self.max_vel
                
                # Integrate position
                x = prev_state.x + vx * self.dt
                y = prev_state.y + vy * self.dt
                
                # Stay within tube (Mahalanobis distance)
                dx_mu = x - mu.x
                dy_mu = y - mu.y
                mahal_dist_sq = (dx_mu**2 / sigma.xx) + (dy_mu**2 / sigma.yy)
                threshold = 9.21  # 99% confidence
                
                if mahal_dist_sq > threshold:
                    scale = np.sqrt(threshold / mahal_dist_sq) * 0.95
                    x = mu.x + dx_mu * scale
                    y = mu.y + dy_mu * scale
                
                rollout.append(State(x, y, vx, vy))
            
            rollouts.append(rollout)
            costs.append(self.evaluate_trajectory(rollout, robot))
        
        # MPPI weighting
        min_cost = min(costs)
        weights = [np.exp(-(c - min_cost) / self.lambda_mppi) for c in costs]
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]
        
        # Compute weighted mean
        mean = []
        for t in range(len(robot.gbp_mu)):
            mean_x = mean_y = mean_vx = mean_vy = 0
            for i, rollout in enumerate(rollouts):
                if len(rollout) > t:
                    mean_x += rollout[t].x * normalized_weights[i]
                    mean_y += rollout[t].y * normalized_weights[i]
                    mean_vx += rollout[t].vx * normalized_weights[i]
                    mean_vy += rollout[t].vy * normalized_weights[i]
            mean.append(State(mean_x, mean_y, mean_vx, mean_vy))
        
        # Compute empirical covariance
        mppi_sigma = []
        for t in range(len(mean)):
            var_x = var_y = 0
            for i, rollout in enumerate(rollouts):
                if len(rollout) > t:
                    dx = rollout[t].x - mean[t].x
                    dy = rollout[t].y - mean[t].y
                    var_x += dx**2 * normalized_weights[i]
                    var_y += dy**2 * normalized_weights[i]
            mppi_sigma.append(Gaussian(var_x + 1e-6, var_y + 1e-6))
        
        uncertainty = [np.sqrt(sig.xx + sig.yy) for sig in mppi_sigma]
        
        return rollouts, normalized_weights, mean, mppi_sigma, uncertainty
    
    def evaluate_trajectory(self, traj: List[State], robot: Robot) -> float:
        """Cost function for trajectory evaluation"""
        cost = 0.0
        
        for t, state in enumerate(traj):
            # Goal cost
            dx = state.x - robot.goal.x
            dy = state.y - robot.goal.y
            goal_dist = np.sqrt(dx**2 + dy**2)
            
            if t == len(traj) - 1:
                cost += goal_dist**2 * 50
            else:
                cost += goal_dist * 2
            
            # Obstacle cost
            for obs in self.obstacles:
                d = self.distance(state, State(obs.x, obs.y)) - obs.radius
                if d < self.safety_dist * 2:
                    penetration = self.safety_dist * 2 - d
                    cost += 2000 * penetration**2
            
            # Collision cost with other robots
            for neighbor_id in robot.neighbors:
                neighbor = self.robots[neighbor_id]
                if len(neighbor.gbp_mu) > t:
                    d = self.distance(state, neighbor.gbp_mu[t])
                    if d < self.safety_dist * 1.5:
                        penetration = self.safety_dist * 1.5 - d
                        cost += 1000 * penetration**2
            
            # Control effort
            cost += (state.vx**2 + state.vy**2) * 0.05
            
            # Smoothness (acceleration)
            if t > 0:
                ax = (state.vx - traj[t-1].vx) / self.dt
                ay = (state.vy - traj[t-1].vy) / self.dt
                cost += (ax**2 + ay**2) * 0.5
            
            # Jerk
            if t > 1:
                ax_curr = (state.vx - traj[t-1].vx) / self.dt
                ay_curr = (state.vy - traj[t-1].vy) / self.dt
                ax_prev = (traj[t-1].vx - traj[t-2].vx) / self.dt
                ay_prev = (traj[t-1].vy - traj[t-2].vy) / self.dt
                jerk_x = (ax_curr - ax_prev) / self.dt
                jerk_y = (ay_curr - ay_prev) / self.dt
                cost += (jerk_x**2 + jerk_y**2) * 0.1
        
        return cost
    
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
    
    def visualize(self, save_to_file=False):
        """Visualize current state"""
        self.ax.clear()
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 600)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0f172a')
        self.fig.patch.set_facecolor('#1e293b')
        
        # Draw obstacles
        for obs in self.obstacles:
            if obs.is_wall:
                circle = Circle((obs.x, obs.y), obs.radius, 
                              color='#475569', ec='#1e293b', linewidth=2)
            else:
                circle = Circle((obs.x, obs.y), obs.radius, 
                              color='#64748b', alpha=0.5, ec='#64748b', linewidth=2)
            self.ax.add_patch(circle)
        
        # Draw for each robot
        for robot in self.robots:
            # GBP mean trajectory (dashed)
            if robot.gbp_mu:
                x_mu = [s.x for s in robot.gbp_mu]
                y_mu = [s.y for s in robot.gbp_mu]
                self.ax.plot(x_mu, y_mu, '--', color=robot.color, 
                           alpha=0.4, linewidth=1)
            
            # GBP uncertainty ellipses
            if robot.gbp_mu and robot.gbp_sigma:
                for t in range(0, len(robot.gbp_mu), 4):
                    mu = robot.gbp_mu[t]
                    sigma = robot.gbp_sigma[t]
                    rx = np.sqrt(sigma.xx) * 2.45  # 95% confidence
                    ry = np.sqrt(sigma.yy) * 2.45
                    ellipse = Ellipse((mu.x, mu.y), 2*rx, 2*ry,
                                    facecolor=robot.color,edgecolor=robot.color, 
                                    linewidth=0.5, alpha=0.15)
                    self.ax.add_patch(ellipse)
            
            # MPPI rollouts (very light)
            if robot.mppi_rollouts:
                for i, rollout in enumerate(robot.mppi_rollouts):
                    weight = robot.mppi_weights[i]
                    alpha = weight * 0.3 + 0.05
                    x = [s.x for s in rollout]
                    y = [s.y for s in rollout]
                    self.ax.plot(x, y, color=robot.color, alpha=alpha, linewidth=0.5)
            
            # Selected trajectory (thick solid)
            if robot.selected_trajectory:
                x_sel = [s.x for s in robot.selected_trajectory]
                y_sel = [s.y for s in robot.selected_trajectory]
                self.ax.plot(x_sel, y_sel, color=robot.color, linewidth=3)
            
            # Trajectory history
            if robot.trajectory_history:
                x_hist = [p[0] for p in robot.trajectory_history]
                y_hist = [p[1] for p in robot.trajectory_history]
                self.ax.plot(x_hist, y_hist, color=robot.color, 
                           alpha=0.6, linewidth=2)
            
            # Goal
            goal_circle = Circle((robot.goal.x, robot.goal.y), 10,
                                facecolor=robot.color, alpha=0.4,
                                edgecolor=robot.color, linewidth=2)
            self.ax.add_patch(goal_circle)
            
            # Current position
            robot_circle = Circle((robot.state.x, robot.state.y), 12,
                                 facecolor=robot.color)
            self.ax.add_patch(robot_circle)
            
            # Direction arrow
            if robot.selected_trajectory and len(robot.selected_trajectory) > 1:
                next_state = robot.selected_trajectory[1]
                self.ax.arrow(robot.state.x, robot.state.y,
                            next_state.x - robot.state.x,
                            next_state.y - robot.state.y,
                            head_width=8, head_length=8, fc='white', ec='white',
                            linewidth=2)
        
        # Title
        self.ax.set_title(f'GBP ↔ MPPI: {self.scenario.replace("_", " ").title()} '
                         f'(Step: {self.time_step})', 
                         color='white', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_to_file:
            plt.savefig(f'frame_{self.time_step:04d}.png', 
                       facecolor='#1e293b', dpi=100)
    
    def run_animation(self, num_steps=200):
        """Run animated simulation"""
        def animate(frame):
            if frame < num_steps:
                self.step()
                self.visualize()
                return []
        
        anim = FuncAnimation(self.fig, animate, frames=num_steps, 
                           interval=100, repeat=False, blit=True)
        plt.show()
        return anim
    
    def run_step_by_step(self, num_steps=1000, pause_time=0.1):
        """Run simulation with step-by-step visualization"""
        plt.ion()
        for step in range(num_steps):
            self.step()
            self.visualize()
            plt.pause(pause_time)
            
            # Check if all robots reached goals
            all_reached = all(
                self.distance(robot.state, robot.goal) < 20 
                for robot in self.robots
            )
            if all_reached:
                print(f"All robots reached goals at step {step}")
                break
        
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    # Create planner
    planner = GBPMPPIPlanner(num_robots=4, scenario='narrow_passage')
    
    # Run simulation
    print("Starting GBP ↔ MPPI planning...")
    print(f"Scenario: {planner.scenario}")
    print(f"Robots: {planner.num_robots}")
    print(f"Obstacles: {len(planner.obstacles)}")
    print("\nPress Ctrl+C to stop")
    
    # Choose one:
    # Option 1: Step-by-step with pause (recommended for debugging)
    # planner.run_step_by_step(num_steps=1000, pause_time=0.01)
    
    # Option 2: Animation (may be slower)
    planner.run_animation(num_steps=1000)