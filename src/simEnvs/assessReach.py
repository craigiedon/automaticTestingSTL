import probRobScene
import pyrep.objects
from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs
from probRobScene.wrappers.coppelia.setupFuncs import top_of
from pyrep import PyRep
from pyrep.objects import Camera
from pyrep.robots.arms.panda import Panda
from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

from src.stl import *


def get_above_object_path(agent: Panda, target_obj: pyrep.objects.Object, z_offset: float = 0.0,
                          ig_cols: bool = False) -> ArmConfigurationPath:
    pos = top_of(target_obj)
    pos[2] += z_offset

    path = agent.get_path(position=pos, euler=[-np.pi, 0.0, np.pi / 2.0], ignore_collisions=ig_cols)  # , euler=orient)
    return path


scenario = probRobScene.scenario_from_file("scenarios/cubeOnTable.prs")
pr = PyRep()
pr.launch(headless=True, responsive_ui=False)

scene_view = Camera('DefaultCamera')
scene_view.set_position([-0.43, 3.4, 2.25])
scene_view.set_orientation(np.array([114, 0.0, 0.0]) * np.pi / 180.0)

ex_world, used_its = scenario.generate()
c_objs = cop_from_prs(pr, ex_world)

cube = c_objs["CUBOID"][0]
initial_cube_pos = np.array(cube.get_position())
panda_1, gripper_1 = Panda(0), PandaGripper(0)

initial_arm_config = panda_1.get_configuration_tree()
initial_arm_joint_pos = panda_1.get_joint_positions()
initial_gripper_config = gripper_1.get_configuration_tree()
initial_gripper_joint_pos = gripper_1.get_joint_positions()


def reset_arm():
    pr.set_configuration_tree(initial_arm_config)
    pr.set_configuration_tree(initial_gripper_config)
    panda_1.set_joint_positions(initial_arm_joint_pos, disable_dynamics=True)
    panda_1.set_joint_target_velocities([0] * 7)
    gripper_1.set_joint_positions(initial_gripper_joint_pos, disable_dynamics=True)
    gripper_1.set_joint_target_velocities([0] * 2)


pr.start()
pr.step()

ts = pr.get_simulation_timestep()
print("timestep:", ts)


def sim_fun(cube_x_guess: float, obj_spec: List[STLExp]) -> float:
    reset_arm()
    new_cube_pos = np.array([cube_x_guess, -0.2, initial_cube_pos[2]])
    cube.set_position(new_cube_pos)

    max_timesteps = 100
    state_information = []

    try:
        arm_path = get_above_object_path(panda_1, cube, 0.03)
        move_done = False
    except Exception as e:
        print(e)
        move_done = True

    for t in range(max_timesteps):
        if not move_done:
            move_done = arm_path.step()

        pr.step()

        target_pos = np.array(top_of(cube)) + np.array([0.0, 0.0, 0.03])
        arm_pos = panda_1.get_tip().get_position()
        state_information.append((target_pos, arm_pos))

    score = agm_rob(obj_spec[0], state_information, 0)
    print("Cube offset: ", cube_x_guess, "score:", score)
    return score


pr.stop()
pr.shutdown()
