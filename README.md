# Cross-domain Multi-task MAMujoco

This repository is an extension of MARL
benchmark [Multi-Agent Mujoco](https://github.com/schroederdewitt/multiagent_mujoco),
and integrates the [DMControl](https://github.com/google-deepmind/dm_control) task settings into MAMujoco.
The rewards of each domain and each task are normalized to [0, 1],
which provides a benchmark for cross-domain multi-task multi-agent reinforcement learning.

这个repo是MARL 基准环境[Multi-Agent Mujoco](https://github.com/schroederdewitt/multiagent_mujoco)的拓展,
将[DMControl](https://github.com/google-deepmind/dm_control)的任务设定集成到了MAMujoco中,
将每个域和每个任务的奖励归一化到[0, 1],
为多智能体系统的跨域多任务强化学习提供benchmark.

## Installation安装 & Start开始
The source code of MAMujoCo has been included in this repository, 
but you still need to install OpenAI gym and mujoco-py support.

这个仓库包含了MAMujoCo的源代码, 
但是仍然需要安装OpenAI gym 和 mujoco_py的支持.

```bash
conda create -n cdmtmujoco python=3.11
conda activate cdmtmujoco
pip install gym, mujoco_py, omegaconf
python example.py
```

The multitasking environment is defined in the class `multitask.MultiTaskMulti`, 
you can modify the `config.json` to confirm which domains and tasks are included in the environment.

多任务的环境在类`multitask.MultiTaskMulti`中定义,
你可以修改`config.json`以确认环境包含哪些域和任务.


## Task Setting 任务设定

<table>
  <tr>
    <th>Suite</th>
    <th>Domain</th>
    <th>Task</th>
    <th>Description</th>
  </tr>
  <tr>
    <td rowspan="2">Ant</td>
    <td rowspan="2"> 2x4 <br> 2x4d </td>
    <td>run</td>
    <td>rewarded by running forward as fast</td>
  </tr>
  <tr>
    <td>run-backwards</td>
    <td>rewarded by running backward as fast</td>
  </tr>
  <tr>
    <td rowspan="5">cheetah</td>
    <td rowspan="5">2x3 <br> 6x1 </td>
    <td>run</td>
    <td>-</td>
  </tr>
  <tr>
    <td>run-backwards</td>
    <td>-</td>
  </tr>
  <tr>
    <td>jump</td>
    <td>rewarded by jumping high and keeping move speed low</td>
  </tr>
  <tr>
    <td>run-front-foot</td>
    <td>rewarded by running speed and the height of behind foot</td>
  </tr>
  <tr>
    <td>run-back-foot</td>
    <td>rewarded by running speed and the height of front foot</td>
  </tr>
  <tr>
    <td rowspan="5">hopper</td>
    <td rowspan="5">3x1</td>
    <td>hop</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hop-backwards</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stand</td>
    <td>rewarded by keeping move speed low and minimize control cost</td>
  </tr>
  <tr>
    <td>flip</td>
    <td>rewarded by flipping angle momentum</td>
  </tr>
  <tr>
    <td>flip-backwards</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="3">humanoid</td>
    <td rowspan="3">9|8</td>
    <td>run</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stand</td>
    <td>-</td>
  </tr>
  <tr>
    <td>walk</td>
    <td>rewarded by keeping move speed in a zone</td>
  </tr>
  <tr>
    <td>humanoid_standup</td>
    <td>9|8</td>
    <td>standup</td>
    <td>rewarded by standing up from a lying position</td>
  </tr>
  <tr>
    <td>reacher</td>
    <td>2</td>
    <td>reach</td>
    <td>rewarded by minimizing the distance between fingertip and target</td>
  </tr>
  <tr>
    <td rowspan="2">swimmer</td>
    <td rowspan="2">2x1</td>
    <td>swim</td>
    <td>-</td>
  </tr>
  <tr>
    <td>swim-backwards</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="5">walker</td>
    <td rowspan="5">2x3</td>
    <td>run</td>
    <td>-</td>
  </tr>
  <tr>
    <td>run-backwards</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stand</td>
    <td>-</td>
  </tr>
  <tr>
    <td>walk</td>
    <td>-</td>
  </tr>
  <tr>
    <td>walk-backwards</td>
    <td>-</td>
  </tr>
</table>

## Reward Normalization 奖励归一化

The reward normalization method in DMControl is applied to this repository, 
and `dm_control/utils/rewards.py` is integrated into the `custom_suites/utils.py`. 
The original task in MAMujoCo does not change the reward design, 
but normalizes the reward to [0, 1]. 
For new tasks, e.g. cheetah-jump reward settings refer to `_jump_reward()` 
in `custom_suites/cheetah.py`.

DMControl中奖励归一化的方法被应用到了这个仓库中, 
`dm_control/utils/rewards.py`被整合到`custom_suites/utils.py`中.
MAMujoCo中原有的任务不改变奖励设计方式, 但是将奖励归一化到[0, 1]之间.
对于新增的任务, 
例如cheetah-jump 奖励设置参考 `custom_suites/cheetah.py`中的`_jump_reward()`.

## Cross-Domain Multi-Task Implement 跨域多任务实现

类`multitask.MultiTaskMulti`中定义了一个列表`self.envs`, 
用于存储所有的属于不同域的环境,
通过`self.reset_task(task_idx)`切换当前任务.
不同域的观测和状态维度不同, 
所以在`_obs_pat(obs)`, `_state_pat(state)`中会被填充到相同的维度, 
以保证环境传递给算法的数据维度一致.
同样, 动作空间的维度和智能体的数量也与最大的任务保持一致, 
与具体环境交互时, `_act_crop(actions)`会根据需要对动作进行裁剪, 
以适应当前任务的要求.

The class `multitask.MultiTaskMulti` defines a list `self.envs`,
Used to store all environments belonging to different domains,
Switch the current task through `reset_task(task_idx)`.
The observation and state dimensions of different domains are different,
so in `_obs_pat(obs)`, `_state_pat(state)` will be filled to the same dimension,
so that the data dimensions passed by the environment to the algorithm are consistent.
Likewise, the dimensions of the action space and the number of agents are consistent with the largest tasks,
when interacting with a specific environment, 
`_act_crop(actions)` will crop the actions as needed,
to adapt to the requirements of the current task.



