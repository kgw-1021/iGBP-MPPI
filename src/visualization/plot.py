import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from matplotlib.animation import FuncAnimation

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

def run_step_by_step(self, num_steps=1000, pause_time=0.01):
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
