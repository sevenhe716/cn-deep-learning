import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)   # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    # 软着陆的奖励函数
    def get_reward2(self, debug=False):
        # Z轴权重最大
        z_penalty = abs(self.sim.pose[2] - self.target_pos[2])

        # X，Y轴给个相对较小的权重
        xy_penalty = abs(self.sim.pose[:2] - self.target_pos[:2]).sum() * 0.5

        # 着陆的速度要足够小
        v_penalty = np.linalg.norm(self.sim.v) * 0.3

        # 着陆的角度尽量平稳
        angle_penalty = abs(self.sim.angular_v[:3]).sum() * 0.25

        total_penalty = z_penalty + xy_penalty + v_penalty + angle_penalty

        reward = 100 - total_penalty

        # 打印奖励的占比，调试用
        if debug:
            print('z={} {}, xy={} {}, v={} {}, angle={} {}'.format(z_penalty, z_penalty / total_penalty, xy_penalty,
                  xy_penalty / total_penalty, v_penalty, v_penalty / total_penalty, angle_penalty, angle_penalty / total_penalty))

        return reward

    # 软着陆的单步函数
    def step2(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward2()
            pose_all.append(self.sim.pose)

            # 着陆后提前终止
            if self.sim.pose[2] <= self.target_pos[2]:
                done = True

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
