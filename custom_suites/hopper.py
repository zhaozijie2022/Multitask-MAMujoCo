import numpy as np

from multiagent_mujoco.mujoco_multi import MujocoMulti
from utils import tolerance

_STAND_HEIGHT = 0.7
_HOP_SPEED = 2.5


class HopperMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)

        self.stand_height = kwargs.get("stand_height", _STAND_HEIGHT)
        self.hop_speed = kwargs.get("hop_speed", _HOP_SPEED)

        self.tasks = ["hop", "hop_backwards", "stand"]
        self.n_tasks = len(self.tasks)
        self._task_idx = 0

    def step(self, actions):
        # we need to map actions back into MuJoCo action space
        env_actions = np.zeros((sum([self.action_space[i].low.shape[0] for i in range(self.n_agents)]),)) + np.nan
        for a, partition in enumerate(self.agent_partitions):
            for i, body_part in enumerate(partition):
                if env_actions[body_part.act_ids] == env_actions[body_part.act_ids]:
                    raise Exception("FATAL: At least one env action is doubly defined!")
                env_actions[body_part.act_ids] = actions[a][i]

        if np.isnan(env_actions).any():
            raise Exception("FATAL: At least one env action is undefined!")

        _, raw_reward, done, info = self.wrapped_env.step(env_actions)
        self.steps += 1

        if done:
            if self.steps < self.episode_limit:
                info["episode_limit"] = False  # the next state will be masked out
            else:
                info["episode_limit"] = True  # the next state will not be masked out

        info["reward_hop"] = raw_reward - 1.0 + 1e-3 * np.square(env_actions).sum()
        info["control"] = env_actions

        # obs_n = self.get_obs()
        # share_obs = self.get_state()
        # share_obs_n = [share_obs[:] for _ in range(self.n_agents)]

        reward = self.get_reward(info)
        # reward_n = [np.array([reward]) for _ in range(self.n_agents)]
        # done_n = [done for _ in range(self.n_agents)]
        # info_n = [info for _ in range(self.n_agents)]
        # available_actions = None
        return reward, done, info

    def reset(self):
        return super().reset()
        # obs_n = super().reset()
        # share_obs = self.get_state()
        # share_obs_n = [share_obs[:] for _ in range(self.n_agents)]
        # available_actions = None
        # return obs_n, share_obs_n, available_actions

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
        speed = info["reward_hop"]
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

    def get_hopper_height(self):
        return self.wrapped_env.env.env.sim.data.qpos[1]

    def get_reward(self, info):
        if self.task == "hop":
            return self._hop_reward(info)
        elif self.task == "hop_backwards":
            return self._hop_backwards_reward(info)
        elif self.task == "stand":
            return self._stand_reward(info)
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
