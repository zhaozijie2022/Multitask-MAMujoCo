import numpy as np

from multiagent_mujoco.mujoco_multi import MujocoMulti
from custom_suites.utils import tolerance

_STAND_HEIGHT = 0.7
_HOP_SPEED = 3.0
_SPIN_SPEED = 5.0


class HopperMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)

        self.stand_height = kwargs.get("stand_height", _STAND_HEIGHT)
        self.hop_speed = kwargs.get("hop_speed", _HOP_SPEED)
        self.spin_speed = kwargs.get("spin_speed", _SPIN_SPEED)

        self.tasks = ["hop", "hop_backwards", "stand", "flip", "flip_backwards"]
        self.n_tasks = len(self.tasks)
        self._task_idx = 0

    def step(self, actions):
        raw_reward, done, info = super().step(actions)
        env_actions = np.concatenate(actions)
        info["reward_hop"] = raw_reward - 1.0 + 1e-3 * np.square(env_actions).sum()
        info["control"] = env_actions
        reward = self.get_reward(info)
        return reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        self.wrapped_env.close()

    def _hop_reward(self, info):
        height = self.get_hopper_height()
        standing = tolerance(height, (self.stand_height, 2))
        speed = info["reward_hop"]
        hopping = tolerance(speed,
                            bounds=(self.hop_speed, float('inf')),
                            margin=self.hop_speed / 2,
                            value_at_margin=0.5,
                            sigmoid='linear')
        return standing * hopping

    def _hop_backwards_reward(self, info):
        height = self.get_hopper_height()
        standing = tolerance(height, (self.stand_height, 2))
        speed = -1.0 * info["reward_hop"]
        hopping = tolerance(speed,
                            bounds=(self.hop_speed / 2, float('inf')),
                            margin=self.hop_speed / 4,
                            value_at_margin=0.5,
                            sigmoid='linear')
        return standing * hopping

    def _stand_reward(self, info):
        height = self.get_hopper_height()
        standing = tolerance(height, (self.stand_height, 2))
        small_control = tolerance(info["control"],
                                  margin=1,
                                  value_at_margin=0,
                                  sigmoid='quadratic')
        small_control = (small_control + 4) / 5
        return standing * small_control

    def _flip_reward(self, info):
        angle_momentum = self.get_hopper_angle_momentum()
        return tolerance(angle_momentum,
                         bounds=(self.spin_speed, float('inf')),
                         margin=self.spin_speed / 2,
                         value_at_margin=0,
                         sigmoid='linear')

    def _flip_backwards_reward(self, info):
        angle_momentum = -1.0 * self.get_hopper_angle_momentum()
        return tolerance(angle_momentum,
                         bounds=(self.spin_speed, float('inf')),
                         margin=self.spin_speed / 2,
                         value_at_margin=0,
                         sigmoid='linear')

    def get_hopper_height(self):
        return self.wrapped_env.env.env.sim.data.qpos[1]

    def get_hopper_angle_momentum(self):
        return self.wrapped_env.env.env.sim.data.subtree_angmom[1][1]

    def get_reward(self, info):
        if self.task == "hop":
            return self._hop_reward(info)
        elif self.task == "hop_backwards":
            return self._hop_backwards_reward(info)
        elif self.task == "stand":
            return self._stand_reward(info)
        elif self.task == "flip":
            return self._flip_reward(info)
        elif self.task == "flip_backwards":
            return self._flip_backwards_reward(info)
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

    def reset_task(self, idx):
        assert 0 <= idx < self.n_tasks
        self._task_idx = idx
        return self.task_idx

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def task(self):
        return self.tasks[self._task_idx]
