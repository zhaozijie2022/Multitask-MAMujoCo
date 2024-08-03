import numpy as np

from multiagent_mujoco.mujoco_multi import MujocoMulti
from custom_suites.utils import tolerance


class ReacherMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)

        geom_names = self.wrapped_env.env.env.model.geom_names
        self.geom_idxes = {name: idx for idx, name in enumerate(geom_names)}

        self.radii = (self.wrapped_env.env.env.model.geom_size[self.geom_idxes["target"]][0]
                      + self.wrapped_env.env.env.model.geom_size[self.geom_idxes["fingertip"]][0])

        self.tasks = ["reach"]
        self.n_tasks = len(self.tasks)
        self._task_idx = 0

    def step(self, actions):
        reward, done, info = super().step(actions)
        reward = self.get_reward(info)
        return reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        self.wrapped_env.close()

    def _reach_reward(self, info):
        return tolerance(-info["reward_dist"],
                         bounds=(0, self.radii))

    def get_finger2target_dist(self):
        f_pos = self.wrapped_env.env.env.sim.data.body_xpos["fingertip"]
        t_pos = self.wrapped_env.env.env.sim.data.body_xpos["target"]
        return np.linalg.norm(f_pos - t_pos)

    def get_reward(self, info):
        if self.task == "reach":
            return self._reach_reward(info)
        else:
            raise NotImplementedError(f"task {self.task} is not implemented.")

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
