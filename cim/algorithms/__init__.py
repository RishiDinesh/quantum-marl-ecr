from .quantum_dqn import get_quantum_dqn, get_quantum_dqn_policy
from .quantum_ppo import get_ppo, get_ppo_policy
from .quantum_maddpg import get_maddpg, get_maddpg_policy
from .quantum_ac import get_ac, get_ac_policy
creators = {
    "quantum_dqn": {
        "trainer": get_quantum_dqn,
        "policy": get_quantum_dqn_policy
    },
    "quantum_ppo": {
        "trainer": get_ppo,
        "policy": get_ppo_policy
    },
    "quantum_maddpg": {
        "trainer": get_maddpg,
        "policy": get_maddpg_policy
    },
    "quantum_ac": {
        "trainer": get_ac,
        "policy": get_ac_policy
    }
}
