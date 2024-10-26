env_conf = {
    "scenario": "cim",
    "topology": "toy.4p_ssdd_l0.0",  # "global_trade.22p_l0.8" for testing
    "durations": 500,
    "start_tick": 0,
    "options": {"enable-dump-snapshot": "./dump_data"}
}

if env_conf["topology"].startswith("toy"):
    num_agents = int(env_conf["topology"].split(".")[1][0])
else:
    num_agents = int(env_conf["topology"].split(".")[1][:2])

port_attributes = ["empty", "full", "on_shipper",
                   "on_consignee", "booking", "shortage", "fulfillment"]
vessel_attributes = ["empty", "full", "remaining_space"]

state_shaping_conf = {
    "look_back": 7,
    "max_ports_downstream": 2,
}

action_shaping_conf = {
    "action_space": [(i - 10) / 10 for i in range(21)],
    "finite_vessel_space": True,
    "has_early_discharge": True,
}

reward_shaping_conf = {
    "time_window": 99,
    "fulfillment_factor": 1.0,
    "shortage_factor": 1.0,
    "time_decay": 0.97,
}

# obtain state dimension from a temporary env_wrapper instance
state_dim = (state_shaping_conf["look_back"] + 1) * (state_shaping_conf["max_ports_downstream"] + 1) * len(
    port_attributes,
) + len(vessel_attributes)

action_num = len(action_shaping_conf["action_space"])

algorithm = "quantum_dqn" # from the algorithms : [quantum_ppo, quantum_dqn, quantum_maddpg, quantum_ac]
