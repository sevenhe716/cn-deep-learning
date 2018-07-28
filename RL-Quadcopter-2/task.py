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
            print('-z={} {}, xy={} {}, v={} {}, angle={} {}'.format(z_penalty, z_penalty / total_penalty, xy_penalty,
                  xy_penalty / total_penalty, v_penalty, v_penalty / total_penalty, angle_penalty, angle_penalty / total_penalty))

        return reward

    # 悬停的奖励函数
    def get_reward3(self, debug=False):
        # 悬停的位置要足够接近
        # pos_penalty = np.linalg.norm(self.sim.pose[:3] - self.target_pos)

        # Z轴权重最大
        z_penalty = abs(self.sim.pose[2] - self.target_pos[2])

        # X，Y轴给个相对较小的权重
        xy_penalty = abs(self.sim.pose[:2] - self.target_pos[:2]).sum() * 0.2

        # 悬停的速度要足够小
        # v_penalty = np.linalg.norm(self.sim.v) * 0.2
        v_penalty = 0

        # 悬停的角度要足够小
        # angle_penalty = abs(self.sim.pose[3:]).sum() * 0.1
        angle_penalty = 0

        # 悬停的角速度尽量小
        # angle_v_penalty = abs(self.sim.angular_v[:3]).sum() * 0.1
        angle_v_penalty = 0

        total_penalty = z_penalty + xy_penalty + v_penalty + angle_penalty + angle_v_penalty

        reward = 100 - total_penalty
        is_reward = False
        reward_value = 0

        # if np.linalg.norm(self.sim.pose[:3] - self.target_pos) < 5:
        # extra bonus
        z_dist = abs(self.sim.pose[2] - self.target_pos[2])
        if z_dist < 80:
            reward_value = (80 - z_dist) * 5
            reward += reward_value
            is_reward = True

        # 打印奖励的占比，调试用
        if debug:
            print('-{} {}, z={} {}, xy={} {}, v={} {}, angle={} {}, angle_v={} {}'.format(is_reward, reward_value, z_penalty, z_penalty / total_penalty,
                                                                                  xy_penalty, xy_penalty / total_penalty, v_penalty,
                                                                          v_penalty / total_penalty, angle_penalty,
                                                                          angle_penalty / total_penalty, angle_v_penalty,
                                                                          angle_v_penalty / total_penalty))

        return reward

    def get_reward4(self):
        distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        sum_acceleration = np.linalg.norm(self.sim.linear_accel)
        reward = (5. - distance_to_target) * 0.3 - sum_acceleration * 0.05
        return reward

    def get_reward5(self):
        """Uses current pose of sim to return reward."""

        # z轴给予较大的奖励
        reward_z = np.tanh(1 - 0.003 * (abs(self.sim.pose[2] - self.target_pos[2]))).sum()
        # xy轴给予较小的奖励
        reward_xy = np.tanh(1 - 0.009 * (abs(self.sim.pose[:2] - self.target_pos[:2]))).sum()

        # 上升的朝向要与初始一致
        # angle_penalty = abs(self.sim.pose[3:]).sum() * 0.001
        reward = reward_z + reward_xy

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
            # if self.sim.pose[2] <= self.target_pos[2]:
            #     done = True

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    # 悬停的单步函数
    def step3(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward3()
            pose_all.append(self.sim.pose)

            # 着陆后提前终止
            # if (self.sim.pose[2] - self.target_pos[2]):
            #     reward += 30
            #     done = True

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def step4(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward4()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def step5(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward5()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
