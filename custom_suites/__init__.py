from .ant import AntMulti
from .cheetah import HalfCheetahMulti
from .hopper import HopperMulti
from .humanoid import HumanoidMulti
from .humanoid_standup import HumanoidStandupMulti
from .reacher import ReacherMulti
from .swimmer import SwimmerMulti
from .walker2d import Walker2dMulti

ENV_REGISTRY = {
    "2_ant": AntMulti, "2_ant_diag": AntMulti, "4_ant": AntMulti,
    "2_cheetah": HalfCheetahMulti, "6_cheetah": HalfCheetahMulti,
    "3_hopper": HopperMulti,
    "2_humanoid": HumanoidMulti,
    "2_humanoid_standup": HumanoidStandupMulti,
    "2_reacher": ReacherMulti,
    "2_swimmer": SwimmerMulti,
    "2_walker2d": Walker2dMulti
}

ARGS_REGISTRY = {
    "2_ant": {"scenario": "Ant-v2", "agent_conf": "2x4", "agent_obsk": 0},
    "2_ant_diag": {"scenario": "Ant-v2", "agent_conf": "2x4d", "agent_obsk": 0},
    "4_ant": {"scenario": "Ant-v2", "agent_conf": "4x2", "agent_obsk": 0},
    "2_cheetah": {"scenario": "HalfCheetah-v2", "agent_conf": "2x3", "agent_obsk": 0},
    "6_cheetah": {"scenario": "HalfCheetah-v2", "agent_conf": "6x1", "agent_obsk": 0},
    "3_hopper": {"scenario": "Hopper-v2", "agent_conf": "3x1", "agent_obsk": 0},
    "2_humanoid": {"scenario": "Humanoid-v2", "agent_conf": "9|8", "agent_obsk": 0},
    "2_humanoid_standup": {"scenario": "HumanoidStandup-v2", "agent_conf": "9|8", "agent_obsk": 0},
    "2_reacher": {"scenario": "Reacher-v2", "agent_conf": "2x1", "agent_obsk": 0},
    "2_swimmer": {"scenario": "Swimmer-v2", "agent_conf": "2x1", "agent_obsk": 0},
    "2_walker2d": {"scenario": "Walker2d-v2", "agent_conf": "2x3", "agent_obsk": 0}
}
