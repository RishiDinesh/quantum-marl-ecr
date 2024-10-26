from functools import partial
from typing import Any, Callable, Dict, List, Optional

from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer
from maro.rl.workflows.callback import Callback
from maro.simulator import Env

from .algorithms import creators
from .config import action_num, algorithm, env_conf, num_agents, reward_shaping_conf, state_dim
from .env_sampler import CIMEnvSampler
from .algorithms import get_quantum_dqn,get_ppo,get_maddpg,get_ac, get_quantum_dqn_policy,get_ppo_policy, get_ac_policy,get_maddpg_policy

class CIMBundle(RLComponentBundle):
    def __init__(self, 
                 env_sampler: AbsEnvSampler=None, 
                 agent2policy: Dict[Any, str]=None, 
                 policies: List[AbsPolicy]=None, 
                 trainers: List[AbsTrainer]=None,
                 device_mapping: Dict[str, str] = None, 
                 policy_trainer_mapping: Dict[str, str] = None, 
                 customized_callbacks: List[Callback] = []
                 ) -> None:
        self.env = Env(scenario="cim", topology=env_conf["topology"], durations=env_conf["durations"])
        self.test_env = None
        self.agent2policy = self.get_agent2policy()
        self.policies = self.get_policies()
        self.trainers = self.get_trainers()
        self.env_sampler = self.get_env_sampler()
        super().__init__(self.env_sampler, self.agent2policy, self.policies, self.trainers, device_mapping, policy_trainer_mapping, customized_callbacks)
       
    def get_env_sampler(self) -> AbsEnvSampler:
        return CIMEnvSampler(self.env, self.test_env,self.policies,self.agent2policy, reward_eval_delay=reward_shaping_conf["time_window"])

    def get_agent2policy(self) -> Dict[Any, str]:
        return {agent: f"{algorithm}_{agent}.policy" for agent in self.env.agent_idx_list}

    def get_policies(self) -> List[AbsPolicy]:
        if algorithm == "quantum_dqn":
            policy_creator = [get_quantum_dqn_policy(state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            ]
        if algorithm == "quantum_ppo":
            policy_creator = [get_ppo_policy(state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            ]
        if algorithm == "quantum_maddpg":
            policy_creator = [get_maddpg_policy(state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            ]
        if algorithm == "quantum_ac":
            policy_creator = [get_ac_policy(state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            ]
        return policy_creator

    def get_trainers(self) -> List[AbsTrainer]:
        if algorithm == "quantum_dqn":
            trainer_creator = [get_quantum_dqn(f"{algorithm}_{i}") for i in range(num_agents)]
        if algorithm == "quantum_ppo":
            trainer_creator = [get_ppo(state_dim,f"{algorithm}_{i}") for i in range(num_agents)]
        if algorithm == "quantum_maddpg":
            trainer_creator = [get_maddpg(state_dim,[1],f"{algorithm}_{i}") for i in range(num_agents)]
        if algorithm == "quantum_ac":
            trainer_creator = [get_ac(state_dim,f"{algorithm}_{i}") for i in range(num_agents)]
        return trainer_creator
