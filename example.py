import json
from omegaconf import OmegaConf, DictConfig
import numpy as np


def main():
    env_cfg = OmegaConf.create(json.load(open("config/config.json", "r")))
    from multitask_mujoco import MultitaskMujoco

    env = MultitaskMujoco(env_cfg)
    for i, mini_env in enumerate(env.envs):
        print(env.tasks[i])
        print(env.envs_info[i])
    n_episodes = env.n_tasks

    for e in range(n_episodes):
        env.reset_task(e % len(env.tasks))
        print(env.task)
        env.reset()
        terminated = [False]
        episode_reward = 0

        while not all(terminated):
            actions = []
            for agent_id in range(env.n_agents):
                # avail_actions = env.get_avail_agent_actions(agent_id)
                # avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.uniform(-1.0, 1.0, env.action_shape)
                actions.append(action)

            # obs_n, reward_n, terminated, info = env.step(actions)
            obs_n, share_obs_n, reward_n, terminated, info_n, avail_actions = env.step(actions)

            episode_reward += reward_n[0]


    env.close()

if __name__ == "__main__":
    main()










