# Set up the drake boilerplate system and vis (maybe just multibody plant without "manipulation station"?)
import time

import numpy as np
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.common.value import Value
from pydrake.geometry import DrakeVisualizer
from pydrake.geometry.render import (MakeRenderEngineVtk, RenderEngineVtkParams)
from pydrake.manipulation.planner import DifferentialInverseKinematicsIntegrator, DifferentialInverseKinematicsParameters
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector, LeafSystem_, BasicVector_, EventStatus
from pydrake.systems.primitives import Integrator, Demultiplexer, Multiplexer, ConstantVectorSource
from pydrake.systems.primitives import StateInterpolatorWithDiscreteDerivative
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp


def make_gripper_position_trajectory(X_G, times):
    """ Constructs a gripper position trajectory from the plan "sketch" """
    tl_ord = sorted(times.keys(), key=lambda k: times[k])
    traj = PiecewisePolynomial.FirstOrderHold(
        [times[tl_ord[0]], times[tl_ord[1]]],
        np.vstack([X_G[tl_ord[0]].translation(), X_G[tl_ord[1]].translation()]).T
    )

    for l in tl_ord[2:]:
        traj.AppendFirstOrderSegment(times[l], X_G[l].translation())

    return traj


def make_gripper_orientation_trajectory(X_G, times):
    """ Constructs a gripper orientation trajectory from the plant "sketch" """
    traj = PiecewiseQuaternionSlerp()
    for label, t in sorted(times.items(), key=lambda kv: kv[1]):
        traj.Append(t, X_G[label].rotation())
    return traj


def make_finger_trajectory(finger_vals, times):
    relevant_times = [k for k, v in times.items() if k in finger_vals]
    tl_ord = sorted(relevant_times, key=lambda k: times[k])
    traj = PiecewisePolynomial.FirstOrderHold(
        [times[tl_ord[0]], times[tl_ord[1]]],
        np.hstack([[finger_vals[tl_ord[0]]],
                   [finger_vals[tl_ord[1]]]])
    )
    for l in tl_ord[2:]:
        traj.AppendFirstOrderSegment(times[l], finger_vals[l])

    return traj


def manual_pick_sketch(X_G_initial, X_O_initial, X_O_goal):
    # Gripper Pose relative to object when in grasp
    p_GgraspO = [0, 0, 0.15]
    R_GgraspO = RotationMatrix.MakeXRotation(np.pi)

    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()

    # Pregrasp is negative z in the gripper frame
    X_GgraspGpregrasp = RigidTransform([0, 0.0, -0.08])

    # TODO: Scoop this part out and feed in (Still need to ensure X_G_initial makes it into key though...)
    X_G = {"initial": X_G_initial}
    X_G["pick_start"] = X_O_initial.multiply(X_OGgrasp)
    X_G["pick_end"] = X_G["pick_start"]
    X_G["prepick"] = X_G["pick_start"].multiply(X_GgraspGpregrasp)
    X_G["postpick"] = X_G["prepick"]
    X_G["place_start"] = X_O_goal.multiply(X_OGgrasp)
    X_G["place_end"] = X_G["place_start"]
    X_G["preplace"] = X_G["place_start"].multiply(X_GgraspGpregrasp)
    X_G["postplace"] = X_G["preplace"]

    # Interpolate a halfway orientation by converting to axis angle and halving angle
    X_GprepickGpreplace = X_G["prepick"].inverse().multiply(X_G["preplace"])
    angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
    X_GprepickGclearance = RigidTransform(AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
                                          X_GprepickGpreplace.translation() / 2.0 + np.array([0, 0.0, -0.5]))
    X_G["clearance"] = X_G["prepick"].multiply(X_GprepickGclearance)

    # Precise timings of trajectory
    times = {"initial": 0}
    X_GinitialGprepick = X_G["initial"].inverse().multiply(X_G["prepick"])
    times["prepick"] = times["initial"] + 10.0 * np.linalg.norm(X_GinitialGprepick.translation())

    # Allow some time for gripper to close
    times["pick_start"] = times["prepick"] + 2.0
    times["pick_end"] = times["pick_start"] + 2.0
    times["postpick"] = times["pick_end"] + 2.0
    time_to_from_clearance = 10.0 * np.linalg.norm(X_GprepickGclearance.translation())
    times["clearance"] = times["postpick"] + time_to_from_clearance
    times["preplace"] = times["clearance"] + time_to_from_clearance
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 2.0
    times["postplace"] = times["place_end"] + 2.0

    opened = np.array([0.08])
    closed = np.array([0.00])
    finger_vals = {"initial": opened,
                   "pick_start": opened,
                   "pick_end": closed,
                   "place_start": closed,
                   "place_end": opened,
                   "postplace": opened}

    pos_traj = make_gripper_position_trajectory(X_G, times)
    rot_traj = make_gripper_orientation_trajectory(X_G, times)
    finger_traj = make_finger_trajectory(finger_vals, times)

    return pos_traj, rot_traj, finger_traj


@TemplateSystem.define("TrajToRB_")
def TrajToRB_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, traj_pos, traj_rot, converter=None):
            LeafSystem_[T].__init__(self, converter=converter)
            self.traj_pos = traj_pos
            self.traj_rot = traj_rot
            self.DeclareAbstractOutputPort("RigidBod", Value[RigidTransform], self.CalcOutput)

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, other.traj_pos, other.traj_rot, converter=converter)

        def CalcOutput(self, context, output):
            t = context.get_time()
            pos_vec = self.traj_pos.value(t)
            rot_mat_vec = self.traj_rot.value(t)

            rb = RigidTransform(Quaternion(rot_mat_vec), pos_vec)
            output.SetFrom(Value[RigidTransform](rb))

    return Impl


@TemplateSystem.define("GripperTrajectoriesToPosition_")
def GripperTrajectoriesToPosition_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, plant, traj_hand, converter=None):
            LeafSystem_[T].__init__(self, converter=converter)
            self.plant = plant
            self.gripper_body = plant.GetBodyByName("panda_hand")
            self.left_finger_joint = plant.GetJointByName("panda_finger_joint1")
            self.right_finger_joint = plant.GetJointByName("panda_finger_joint2")
            self.traj_hand = traj_hand
            self.plant_context = plant.CreateDefaultContext()

            self.DeclareVectorOutputPort("finger_position", BasicVector_[T](2), self.CalcPositionOutput)

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, other.plant, other.traj_hand, converter=converter)

        def CalcPositionOutput(self, context, output):
            t = context.get_time()
            hand_command = self.traj_hand.value(t)
            self.left_finger_joint.set_translation(self.plant_context, hand_command / 2.0)
            self.right_finger_joint.set_translation(self.plant_context, hand_command / 2.0)
            output.SetFromVector(self.plant.GetPositions(self.plant_context)[-2:])

    return Impl


def add_named_system(builder, name, system):
    """ Although the Drake docs *say* that DiagramBuilder.AddNamedSystem is supported in the python bindings,
    that does not appear to be true. So i've implemented it here"""
    s = builder.AddSystem(system)
    s.set_name(name)
    return s


def inverse_dynamics_standard(controller_plant: MultibodyPlant):
    kp = np.full(9, 100)
    ki = np.full(9, 1)
    kd = 2 * np.sqrt(kp)
    return InverseDynamicsController(controller_plant, kp, ki, kd, False)


class DifferentialIKSystem(LeafSystem):
    def __init__(self, plant, diff_ik_func):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()

        self._panda = plant.GetModelInstanceByName("panda")

        self.panda_start = plant.GetJointByName("panda_joint1").velocity_start()
        self.panda_end = self.panda_start + 8  # TODO: Make this more robust/flexible

        self._G = plant.GetBodyByName("panda_hand").body_frame()
        self._W = plant.world_frame()
        self._diff_ik_func = diff_ik_func

        self.DeclareVectorInputPort("desired_spatial_vel", BasicVector(6))
        self.DeclareVectorInputPort("current_pos", BasicVector(9))
        self.DeclareVectorInputPort("estimated_vel", BasicVector(9))
        self.DeclareVectorOutputPort("commanded_vel", BasicVector(9), self.CalcOutput)

    def CalcOutput(self, context, output):
        V_G_desired = self.GetInputPort("desired_spatial_vel").Eval(context)
        q_now = self.GetInputPort("current_pos").Eval(context)
        v_now = self.GetInputPort("estimated_vel").Eval(context)

        self._plant.SetPositions(self._plant_context, self._panda)
        J_G = self._plant.CalcJacobianSpatialVelocity(self._plant_context, JacobianWrtVariable.kQDot, self._G, [0, 0, 0], self._W, self._W)
        J_G = J_G[:, self.panda_start:self.panda_end + 1]  # Question: Am i now keeping the gripper terms around?

        X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)
        p_now = X_now.translation()

        v = self._diff_ik_func(J_G, V_G_desired, q_now, v_now, p_now)
        output.SetFromVector(v)


class FrameTracker(LeafSystem):
    def __init__(self, plant, frame_name):
        LeafSystem.__init__(self)
        self.tracked_frame = plant.GetFrameByName(frame_name)
        self._plant = plant
        self.DeclareVectorOutputPort("frame_world_pos", BasicVector(3), self.CalcOutput)

    def CalcOutput(self, context, output):
        frame_world_trans = self.tracked_frame.CalcPoseInWorld(context).translation()
        output.SetFromVector(frame_world_trans)


def panda_constrained_controller(V_d, diff_ik_func, panda, panda_plant):
    b = DiagramBuilder()
    ts = 1e-3

    diff_ik_controller = b.AddSystem(DifferentialIKSystem(panda_plant, diff_ik_func))
    integrator = b.AddSystem(Integrator(9))
    inv_d = b.AddSystem(inverse_dynamics_standard(panda_plant))
    est_state_vel_demux = b.AddSystem(Demultiplexer(np.array([9, 9])))
    des_state_vel_mux = b.AddSystem(Multiplexer(np.array([9, 9])))
    desired_vel_source = b.AddSystem(ConstantVectorSource(V_d))

    b.Connect(panda_plant.get_state_output_port(panda), est_state_vel_demux.get_input_port())

    b.Connect(desired_vel_source.get_output_port(), diff_ik_controller.GetInputPort("desired_spatial_vel"))
    b.Connect(est_state_vel_demux.get_output_port(0), diff_ik_controller.GetInputPort("current_pos"))
    b.Connect(est_state_vel_demux.get_output_port(1), diff_ik_controller.GetInputPort("estimated_vel"))

    b.Connect(diff_ik_controller.GetOutputPort("commanded_vel"), integrator.get_input_port())
    b.Connect(integrator.get_output_port(), des_state_vel_mux.get_input_port(0))
    b.Connect(diff_ik_controller.GetOutputPort("commanded_vel"), des_state_vel_mux.get_input_port(1))

    b.Connect(panda_plant.get_state_output_port(panda), inv_d.get_input_port_estimated_state())
    b.Connect(des_state_vel_mux.get_output_port(), inv_d.get_input_port_desired_state())

    b.ExportOutput(inv_d.get_output_port_control())

    diagram = b.Build()
    return diagram


def panda_traj_controller(traj_pos, traj_rot, traj_hand, panda_plant):
    b = DiagramBuilder()
    ts = 1e-3

    ### Add Systems
    traj_to_rigid = add_named_system(b, "RB Conv", TrajToRB_[None](traj_pos, traj_rot))

    hand_frame = panda_plant.GetFrameByName("panda_hand", control_only_panda)
    ik_params = DifferentialInverseKinematicsParameters(num_positions=9, num_velocities=9)
    ik = add_named_system(b, "Inverse Kinematics", DifferentialInverseKinematicsIntegrator(panda_plant, hand_frame, ts, ik_params))

    diff_arm_demux = add_named_system(b, "Diff Arm Demux", Demultiplexer(np.array([7, 2])))
    arm_hand_mux = add_named_system(b, "Arm-Hand Mux", Multiplexer(np.array([7, 2])))
    s_interp = add_named_system(b, "State Interp", StateInterpolatorWithDiscreteDerivative(9, ts, True))
    hand_comms = add_named_system(b, "GripperTraj", GripperTrajectoriesToPosition_[None](panda_plant, traj_hand))

    kp = np.full(9, 100)
    ki = np.full(9, 1)
    kd = 2 * np.sqrt(kp)
    inv_d = add_named_system(b, "Inverse Dynamics", InverseDynamicsController(panda_plant, kp, ki, kd, False))

    traj_to_rigid.ToAutoDiffXd()

    ### Connect Systems Together
    b.ExportInput(inv_d.get_input_port_estimated_state())

    b.Connect(traj_to_rigid.get_output_port(), ik.get_input_port())
    b.Connect(ik.get_output_port(), diff_arm_demux.get_input_port())
    b.Connect(diff_arm_demux.get_output_port(0), arm_hand_mux.get_input_port(0))
    b.Connect(hand_comms.get_output_port(), arm_hand_mux.get_input_port(1))
    b.Connect(arm_hand_mux.get_output_port(), s_interp.get_input_port())
    b.Connect(s_interp.get_output_port(), inv_d.GetInputPort("desired_state"))

    b.ExportOutput(inv_d.get_output_port_control())
    diagram = b.Build()
    diagram.set_name("Panda Traj Controller")
    return diagram


builder = DiagramBuilder()


class DifferentialIKSystem(LeafSystem):
    def __init__(self, plant, diff_ik_func):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()

        self._panda = plant.GetModelInstanceByName("panda")

        self.panda_start = plant.GetJointByName("panda_joint1").velocity_start()
        self.panda_end = self.panda_start + 8  # TODO: Make this more robust/flexible

        self._G = plant.GetBodyByName("panda_hand").body_frame()
        self._W = plant.world_frame()
        self._diff_ik_func = diff_ik_func

        self.DeclareVectorInputPort("desired_spatial_vel", BasicVector(6))
        self.DeclareVectorInputPort("current_pos", BasicVector(9))
        self.DeclareVectorInputPort("estimated_vel", BasicVector(9))
        self.DeclareVectorOutputPort("commanded_vel", BasicVector(9), self.CalcOutput)

    def CalcOutput(self, context, output):
        V_G_desired = self.GetInputPort("desired_spatial_vel").Eval(context)
        q_now = self.GetInputPort("current_pos").Eval(context)
        v_now = self.GetInputPort("estimated_vel").Eval(context)

        self._plant.SetPositions(self._plant_context, self._panda, q_now)
        J_G = self._plant.CalcJacobianSpatialVelocity(self._plant_context, JacobianWrtVariable.kQDot, self._G, [0, 0, 0], self._W, self._W)
        J_G = J_G[:, self.panda_start:self.panda_end + 1]  # Question: Am i now keeping the gripper terms around?

        X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)
        p_now = X_now.translation()

        v = self._diff_ik_func(J_G, V_G_desired, q_now, v_now, p_now)
        output.SetFromVector(v)


# A multibody plant is a *system* (in the drake sense) holding all the info for multi-body rigid bodies, providing ports for forces and continuous state, geometry etc.
# The "builder" as an argument is a kind of a "side-effectey" thing, we want to add our created multibodyplant *to* this builder
# The 0.0 is the "discrete time step". A value of 0.0 means that we have made this a continuous system
# Note also, this constructor is overloaded (but this is not a thing you can do in python naturally. It is an artefact of the C++ port)
time_step = 0.002
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)

# Load in the panda
# The parser takes a multibody plant mutably, so that anything parsed with it gets automatically added to this multibody system
parser = Parser(plant)

# Files are getting grabbed from "/opt/drake/share/drake/...
panda_arm_hand_file = FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf")
brick_file = FindResourceOrThrow("drake/examples/manipulation_station/models/061_foam_brick.sdf")
bin_file = FindResourceOrThrow("drake/examples/manipulation_station/models/bin.sdf")

# Actually parse in the model
panda = parser.AddModelFromFile(panda_arm_hand_file, model_name="panda")

# Don't want panda to drop through the sky or fall over so...
# It would be nice if there was an option to create a floor...
# The X_AB part is "relative pose" (monogram notation: The pose (X) of B relative to A
plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("panda_link0", panda), X_PC=RigidTransform(np.array([0.0, 0.5, 0.0]))
)

# Add a little example brick
brick = parser.AddModelFromFile(brick_file, model_name="brick")
brick_initial = RigidTransform([0.65, 0.5, 0.015])
# plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link"))

# Add some bins
bin_1 = parser.AddModelFromFile(bin_file, model_name="bin_1")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("bin_base", bin_1))

bin_2 = parser.AddModelFromFile(bin_file, model_name="bin_2")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("bin_base", bin_2), X_PC=RigidTransform(np.array([0.65, 0.5, 0.0])))

# Important to do tidying work
plant.Finalize()

temp_context = plant.CreateDefaultContext()
temp_plant_context = plant.GetMyContextFromRoot(temp_context)
desired_initial_state = np.array([0.0, 0.3, 0.0, -1.3, 0.0, 1.65, 0.9, 0.040, 0.040])
plant.SetPositions(temp_plant_context, panda, desired_initial_state)

traj_p_G, traj_R_G, traj_h = manual_pick_sketch(plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("panda_hand")),
                                                brick_initial,
                                                RigidTransform(RotationMatrix.MakeZRotation(np.pi / 2.0), [0.0, 0.0, 0.015]))

controller_plant = MultibodyPlant(time_step)
controller_parser = Parser(controller_plant)
control_only_panda = controller_parser.AddModelFromFile(panda_arm_hand_file)
controller_plant.WeldFrames(controller_plant.world_frame(), controller_plant.GetFrameByName("panda_link0", control_only_panda), X_PC=RigidTransform(np.array([0.0, 0.5, 0.0])))
controller_plant.Finalize()

traj_controller = builder.AddSystem(panda_traj_controller(traj_p_G, traj_R_G, traj_h, controller_plant))
builder.Connect(plant.get_state_output_port(panda), traj_controller.get_input_port())
builder.Connect(traj_controller.get_output_port(), plant.get_actuation_input_port())
ik_subsystem = traj_controller.GetSubsystemByName("Inverse Kinematics")

scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
dv = DrakeVisualizer.AddToBuilder(builder, scene_graph)
dv.set_name("visuals")

diagram = builder.Build()
diagram.set_name("pick_and_place")

for i in np.linspace(0.02, 0.07, num=500):
    simulator = Simulator(diagram)
    sim_context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(sim_context)
    plant.SetPositions(plant_context, panda, desired_initial_state)
    brick_body = plant.GetBodyByName("base_link", brick)
    brick_body.SetMass(plant_context, i)

    print("Brick mass: ", brick_body.get_mass(plant_context))

    plant.SetFreeBodyPose(plant_context, brick_body, brick_initial)
    hand_frame = plant.GetFrameByName("panda_hand")
    ik_subsystem.SetPositions(ik_subsystem.GetMyMutableContextFromRoot(simulator.get_mutable_context()), plant.GetPositions(plant_context, panda))

    state_traj = []
    def monitor_func(c):
        hand_translation = hand_frame.CalcPoseInWorld(plant_context).translation()
        brick_translation = brick_body.EvalPoseInWorld(plant_context).translation()
        state_traj.append((list(hand_translation), list(brick_translation)))
        # print(f"Hand: {hand_translation}, brick: {brick_translation}")
        return EventStatus.Succeeded()


    simulator.set_monitor(monitor_func)
    simulator.Initialize()
    start_time = time.time()
    simulator.AdvanceTo(traj_p_G.end_time())

    print(f"Time Taken: {time.time() - start_time}")