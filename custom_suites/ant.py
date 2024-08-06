import numpy as np

from multiagent_mujoco.mujoco_multi import MujocoMulti
from custom_suites.utils import tolerance

_RUN_SPEED = 6
_RUN_BACKWARDS_SPEED = 4


class AntMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)

        self.run_speed = kwargs.get("run_speed", _RUN_SPEED)
        self.run_backwards_speed = kwargs.get("run_backwards_speed", _RUN_BACKWARDS_SPEED)
        self.tasks = ["run", "run_backwards"]
        self.n_tasks = len(self.tasks)
        self._task_idx = 0

    def step(self, actions):
        reward, done, info = super().step(actions)
        env_actions = np.concatenate(actions)
        info["control"] = env_actions
        reward = self.get_reward(info)
        return reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        self.wrapped_env.close()

    def _run_reward(self, info):
        speed_reward = tolerance(info["reward_forward"],
                                 bounds=(self.run_speed, float('inf')),
                                 margin=self.run_speed,
                                 value_at_margin=0,
                                 sigmoid='linear')
        control_reward = tolerance(info["control"],
                                   margin=1,
                                   value_at_margin=0,
                                   sigmoid='quadratic').mean()
        return (2 * control_reward + 3 * speed_reward) / 5


    def _run_backwards_reward(self, info):
        speed_reward = tolerance(-1.0 * info["reward_forward"],
                                 bounds=(self.run_backwards_speed, float('inf')),
                                 margin=self.run_backwards_speed,
                                 value_at_margin=0,
                                 sigmoid='linear')
        control_reward = tolerance(info["control"],
                                   margin=1,
                                   value_at_margin=0,
                                   sigmoid='quadratic').mean()
        return (2 * control_reward + 3 * speed_reward) / 5


    def get_reward(self, info):
        if self.task == "run":
            return self._run_reward(info)
        elif self.task == "run_backwards":
            return self._run_backwards_reward(info)
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
