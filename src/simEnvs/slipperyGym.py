import itertools
import pickle

import dill
import numpy as np
from sb3_contrib import TQC

model_path = "learned_models/FetchSlide-v1/FetchSlide-v1.zip"
args_path = "learned_models/FetchSlide-v1/args.yml"

with open("learned_models/saved_fetch_slippy_env", "rb") as f:
    env = pickle.load(f)

with open("learned_models/FetchSlide-v1/saved_fetch_model_custom_objects", "rb") as f:
    custom_objects = dill.load(f)

kwargs = {'seed': 0, 'buffer_size': 1}
pre_trained = TQC.load(model_path, env=env, custom_objects=custom_objects, **kwargs)

inner_env = env.envs[0].env.env.env

g_num_per_dim = 10
g_range = np.linspace(-inner_env.target_range, inner_env.target_range, num=g_num_per_dim)
goals = np.array(list(itertools.product(g_range, g_range)))

puck_offsets = np.array([[0.0, 0.1]])
for gi in range(len(goals)):
    print("Episode: ", gi)
    obs = env.reset()

    # Goal state randomization
    grip_start = inner_env.initial_gripper_xpos[:3]
    new_goal = grip_start + np.append(goals[gi], 0.0) + inner_env.target_offset
    new_goal[2] = inner_env.height_offset
    inner_env.goal = new_goal

    # Puck start state setting
    puck_new_xy_pos = grip_start[:2] + puck_offsets[0]
    # puck_new_xy_pos = np.array([1.3244 + 0.625, 0.7501 - 0.45])
    offnorm = np.linalg.norm(grip_start[:2] - puck_new_xy_pos)
    # assert offnorm >= 0.1
    puck_qpos = inner_env.sim.data.get_joint_qpos("object0:joint")
    puck_qpos[:2] = puck_new_xy_pos
    inner_env.sim.data.set_joint_qpos("object0:joint", puck_qpos)
    inner_env.sim.forward()

    obs, _, _, _ = env.step(np.zeros(4))
    state = None
    for _ in range(50):
        # env.render()
        action, state = pre_trained.predict(obs, state=state, deterministic=True)
        obs, reward, state, infos = env.step(action)


env.reset()
env.close()
