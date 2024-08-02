import numpy as np
import yaml
from gym.spaces import Box

from multiagent_mujoco.mujoco_multi import MujocoMulti


def get_env_args(env_names):
    tasks, env_args_list = [], []
    all_env_args = yaml.load(open("envs/mujoco/env_args.yaml", "r"), Loader=yaml.FullLoader)
    for _env in env_names:
        assert _env["env_name"] in all_env_args, "env_name %s not found" % _env["env_name"]
        tasks.append(_env["env_name"])

        env_args = all_env_args[_env["env_name"]]["env_args"]
        env_args["episode_limit"] = _env["episode_limit"]
        env_args_list.append(env_args)
    return tasks, env_args_list


def make_multitask_envs(env_args_list):
    envs = []
    for env_args in env_args_list:
        env = MujocoMulti(env_args=env_args)
        envs.append(env)
    return envs


def get_action_shape_n(envs_info):
    multitask_action_shape_n = []
    for env_info in envs_info:
        multitask_action_shape_n.append([])
        for action_space in env_info["action_spaces"]:
            multitask_action_shape_n[-1].append(action_space.shape[0])
    return multitask_action_shape_n


class MultitaskMujoco:
    def __init__(self, cfg):
        self.cfg = cfg

        self.max_ep_len = cfg.max_ep_len
        self.tasks, self.env_args_list = get_env_args(cfg.env_names)
        self.envs = make_multitask_envs(self.env_args_list)
        self.envs_info = [env.get_env_info() for env in self.envs]

        self.obs_size = max([env_info["obs_shape"] for env_info in self.envs_info])
        self.share_obs_size = max([env_info["state_shape"] for env_info in self.envs_info])
        self.action_shape = max([env_info["n_actions"] for env_info in self.envs_info])
        self.multitask_action_shape_n = get_action_shape_n(self.envs_info)

        self.agent_nums = [env_info["n_agents"] for env_info in self.envs_info]

        self.n_agents = max(self.agent_nums)

        self.observation_space = [Box(low=np.array([-10] * self.obs_size, dtype=np.float32),
                                      high=np.array([10] * self.obs_size, dtype=np.float32),
                                      dtype=np.float32) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=np.array([-10] * self.share_obs_size, dtype=np.float32),
                                            high=np.array([10] * self.share_obs_size, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]
        self.action_space = tuple([Box(low=np.array([-1] * self.action_shape, dtype=np.float32),
                                       high=np.array([1] * self.action_shape, dtype=np.float32),
                                       dtype=np.float32) for _ in range(self.n_agents)])

        self._task_idx = 0
        self._task = self.tasks[0]
        self.env = self.envs[0]

    def reset(self):
        obs_n = self._obs_pat(self.env.reset())
        share_obs_n = [self._state_pat(self.env.get_state()) for _ in range(len(obs_n))]
        available_actions = None
        return obs_n, share_obs_n, available_actions

    def step(self, actions):
        actions_env = self._act_crop(actions)

        reward, done, info = self.env.step(actions_env)

        obs_n = self._obs_pat(self.env.get_obs())
        share_obs_n = [self._state_pat(self.env.get_state()) for _ in range(len(obs_n))]

        reward_n = [np.array([reward]) for _ in range(len(obs_n))]
        done_n = [done for _ in range(len(obs_n))]
        info["task"] = self.task
        info["task_idx"] = self.task_idx
        info["bad_transition"] = False
        info_n = [info for _ in range(len(obs_n))]

        available_actions = None
        return obs_n, share_obs_n, reward_n, done_n, info_n, available_actions

    def reset_task(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(len(self.tasks))
        self._task_idx = task_idx
        self._task = self.tasks[self._task_idx]
        self.env = self._env
        return self.task_idx

    def close(self):
        for env in self.envs:
            env.close()

    @property
    def task(self):
        return self._task

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def _env(self):
        return self.envs[self.task_idx]

    @property
    def steps(self):
        return self.envs[self.task_idx].steps

    def _act_crop(self, actions):
        # 裁剪actions, 根据multitask_action_shape_n
        actions_env = [
            actions[i, :self.multitask_action_shape_n[self.task_idx][i]]
            for i in range(self.agent_nums[self.task_idx])
        ]
        # for i in range(self.agent_nums[self.task_idx]):
        #     actions[i] = actions[i][:self.multitask_action_shape_n[self.task_idx][i]]
        # 只保留当前env用到的agent的actions
        return actions_env

    def _obs_pat(self, obs_n):
        # 填充obs_n,
        for i, obs in enumerate(obs_n):
            obs_n[i] = np.concatenate([obs, np.zeros(self.obs_size - len(obs))])
        for _ in range(len(obs_n), self.n_agents):
            obs_n.append(np.zeros(self.obs_size))
        return obs_n

    def _state_pat(self, state):
        return np.concatenate([state, np.zeros(self.share_obs_size - len(state))])
