# Cross-domain Multi-task MAMujoco

This repository integrates and extends the [Multi-Agent Mujoco](https:github.comschroederdewittmultiagent_mujoco),
and integrates the [DMControl](https:github.comgoogle-deepminddm_control) tasks into MAMujoco. 
The reward of cross-domain multi-task is normalized to [0, 1], 
which provides a benchmark for cross-domain multi-task multi-agent reinforcement learning.

这个仓库将[Multi-Agent Mujoco](https://github.com/schroederdewitt/multiagent_mujoco)进行了整合和拓展, 
将[DMControl](https://github.com/google-deepmind/dm_control)的任务集集成到了MAMujoco中,
将跨域多任务的奖励归一化到[0, 1],
为多智能体系统的跨域多任务强化学习提供benchmark.

### 任务设定

<table>
  <tr>
    <th>Suite</th>
    <th>Domain</th>
    <th>Task</th>
  </tr>
  <tr>
    <td rowspan="2">Ant</td>
    <td rowspan="2"> 2x4 <br> 2x4d </td>
    <td>run</td>
  </tr>
  <tr>
    <td>run-backwards</td>
  </tr>
  <tr>
    <td rowspan="5">cheetah</td>
    <td rowspan="5">2x3 <br> 6x1 </td>
    <td>run</td>
  </tr>
  <tr>
    <td>run-backwards</td>
  </tr>
  <tr>
    <td>jump</td>
  </tr>
  <tr>
    <td>run-front-foot</td>
  </tr>
  <tr>
    <td>run-back-foot</td>
  </tr>
  <tr>
    <td rowspan="5">hopper</td>
    <td rowspan="5">3x1</td>
    <td>hop</td>
  </tr>
  <tr>
    <td>hop-backwards</td>
  </tr>
  <tr>
    <td>stand</td>
  </tr>
  <tr>
    <td>flip</td>
  </tr>
  <tr>
    <td>flip-backwards</td>
  </tr>
  <tr>
    <td rowspan="3">humanoid</td>
    <td rowspan="3">9|8</td>
    <td>run</td>
  </tr>
  <tr>
    <td>stand</td>
  </tr>
  <tr>
    <td>walk</td>
  </tr>
  <tr>
    <td>humanoid_standup</td>
    <td>9|8</td>
    <td>standup</td>
  </tr>
  <tr>
    <td>reacher</td>
    <td>2</td>
    <td>reach</td>
  </tr>
  <tr>
    <td rowspan="2">swimmer</td>
    <td rowspan="2">2x1</td>
    <td>swim</td>
  </tr>
  <tr>
    <td>swim-backwards</td>
  </tr>
  <tr>
    <td rowspan="5">walker</td>
    <td rowspan="5">2x3</td>
    <td>run</td>
  </tr>
  <tr>
    <td>run-backwards</td>
  </tr>
  <tr>
    <td>stand</td>
  </tr>
  <tr>
    <td>walk</td>
  </tr>
  <tr>
    <td>walk-backwards</td>
  </tr>
</table>

### 奖励归一化

### 跨域多任务实现
