import numpy as np
import copy
from core.types import State, Gaussian
from core.robot import Robot
from utils.utils import distance
from typing import List, Tuple

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