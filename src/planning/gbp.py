import copy
import numpy as np
from typing import List, Tuple
from core.types import State, Gaussian
from core.robot import Robot

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