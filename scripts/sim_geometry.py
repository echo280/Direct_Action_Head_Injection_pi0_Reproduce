"""Oracle geometry used by replay and ID rollout. Requires the simulator environment."""
import numpy as np
from common import simulator_setup

simulator_setup()
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import camera_utils
from openpi.spatial.geometry import innermost_pixel, lift_pixel, project_point


def make_env(bddl):
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                             camera_widths=256, camera_depths=True, camera_segmentations="instance")


def goal_pair(env):
    goals = env.env.parsed_problem["goal_state"]
    pairs = [goal for goal in goals if len(goal) == 3 and goal[0].lower() in ("on", "in")]
    if len(pairs) != 1:
        raise ValueError(f"Expected one grasp/place goal, got {goals}")
    return pairs[0][1], pairs[0][2]


def world_to_base(env):
    sim = env.env.sim
    name = env.env.robots[0].robot_model.root_body
    body = sim.model.body_name2id(name)
    base_to_world = np.eye(4)
    base_to_world[:3, :3] = sim.data.body_xmat[body].reshape(3, 3)
    base_to_world[:3, 3] = sim.data.body_xpos[body]
    return np.linalg.inv(base_to_world)


def object_height(env, name):
    world = env.env.object_states_dict[name].get_geom_state()["pos"]
    return float((world_to_base(env) @ np.r_[world, 1])[2])


def geometry(env, obs, active_target):
    sim = env.env.sim
    depth = camera_utils.get_real_depth_map(sim, obs["agentview_depth"].squeeze())
    k = camera_utils.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
    to_base = world_to_base(env)
    camera_to_base = to_base @ camera_utils.get_camera_extrinsic_matrix(sim, "agentview")
    instances = list(env.env.model.instances_to_ids.keys())
    if active_target in instances:
        mask = obs["agentview_segmentation_instance"].squeeze() == instances.index(active_target) + 1
        uv = innermost_pixel(mask)
    else:
        # Abstract region: project its simulator center, then re-lift visible depth.
        world = env.env.object_states_dict[active_target].get_geom_state()["pos"]
        center = (to_base @ np.r_[world, 1])[:3]
        uv = np.floor(project_point(center, k, camera_to_base) + 0.5).astype(int)
    target = lift_pixel(uv, depth, k, camera_to_base)
    robot = env.env.robots[0]
    gripper_world = sim.data.site_xpos[robot.eef_site_id]
    gripper = (to_base @ np.r_[gripper_world, 1])[:3].astype(np.float32)
    return {"displacement": target - gripper, "target_base": target, "gripper_base": gripper,
            "target_uv": uv.astype(np.int32), "intrinsics": k.astype(np.float32),
            "camera_to_base": camera_to_base.astype(np.float32)}


def model_images(obs):
    # cv observation already flipped vertically; hflip matches official LIBERO
    # VLA preprocessing of default OpenGL observations by a 180-degree rotation.
    return (np.ascontiguousarray(obs["agentview_image"][:, ::-1]),
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][:, ::-1]))


def model_state(obs):
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)
