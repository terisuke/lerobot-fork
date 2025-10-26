#!/usr/bin/env python3
"""
Cup Manipulation Demonstration Recording for SO101
Based on official examples/so100_to_so100_EE/record.py

Uses:
- SO100Leader arm for teleoperation
- SO100Follower arm for task execution
- Webcam for wider FOV (1280x720)
- Proper dataset format for ACT training
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Configuration
NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 120  # 2 minutes per demo
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Pick up the cup and place it next to the bottle"
HF_REPO_ID = "teradakousuke/cup_to_bottle_dataset_v2"

# Camera configuration - Use actual camera capabilities
# Camera 0: 1280x960 @ 25 FPS (confirmed)
camera_config = {
    "front": OpenCVCameraConfig(
        index_or_path=0,
        fps=30.0,  # Match recording FPS
        width=640,  # Reduce for processing speed
        height=480
    )
}

# Robot configurations - Using your actual working setup
follower_config = SO101FollowerConfig(
    port="/dev/tty.wchusbserial5AB90691861",  # Your follower arm
    id="my_awesome_follower_arm",
    cameras=camera_config,
    use_degrees=False,  # Match your working teleoperate command
)

leader_config = SO100LeaderConfig(
    port="/dev/tty.wchusbserial5AB90684961",  # Your leader arm
    id="my_awesome_leader_arm",
    # No use_degrees for leader
)

print("=" * 70)
print("Cup Manipulation Demonstration Recording")
print("=" * 70)
print(f"Task: {TASK_DESCRIPTION}")
print(f"Episodes: {NUM_EPISODES}")
print(f"Follower: {follower_config.port}")
print(f"Leader: {leader_config.port}")
print("=" * 70)
print()

# Initialize the robot and teleoperator
print("Initializing robot and teleoperator...")
follower = SO101Follower(follower_config)
leader = SO100Leader(leader_config)

# Kinematics solver (optional but recommended)
# Download from: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
URDF_PATH = "./SO101/so101_new_calib.urdf"
follower_kinematics_solver = None
leader_kinematics_solver = None

try:
    import placo  # Check if placo is available
    import os
    if os.path.exists(URDF_PATH):
        follower_kinematics_solver = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="gripper_frame_link",
            joint_names=list(follower.bus.motors.keys()),
        )
        leader_kinematics_solver = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="gripper_frame_link",
            joint_names=list(leader.bus.motors.keys()),
        )
        print(f"✅ Kinematics solver loaded from {URDF_PATH}")
    else:
        print(f"⚠️  URDF file not found at {URDF_PATH}")
        print("   Download from: https://github.com/TheRobotStudio/SO-ARM100")
except (ImportError, FileNotFoundError) as e:
    print(f"⚠️  Kinematics solver not available: {e}")
    print("   Continuing without kinematics (will use direct joint control)")

# Build pipeline to convert follower joints to EE observation
if follower_kinematics_solver is not None:
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Build pipeline to convert leader joints to EE action
    leader_joints_to_ee = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver,
                motor_names=list(leader.bus.motors.keys()),
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Build pipeline to convert EE action to follower joints
    ee_to_follower_joints = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
else:
    follower_joints_to_ee = None
    leader_joints_to_ee = None
    ee_to_follower_joints = None

# Create the dataset
print("\nCreating dataset...")
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features=combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=leader_joints_to_ee or RobotProcessorPipeline(),
            initial_features=create_initial_features(action=leader.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=follower_joints_to_ee or RobotProcessorPipeline(),
            initial_features=create_initial_features(
                observation=follower.observation_features
            ),
            use_videos=True,
        ),
    ),
    robot_type=follower.name,
    use_videos=True,
    image_writer_threads=4,
)
print("✅ Dataset created")

# Connect the robot and teleoperator
print("\nConnecting to robot and teleoperator...")
# Skip calibration if already calibrated
leader.connect(calibrate=False)
follower.connect(calibrate=False)
print("✅ Connected")

# Initialize keyboard listener and visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="recording_cup_demos")

if not leader.is_connected or not follower.is_connected:
    raise ValueError("❌ Robot or teleoperator is not connected!")

print("\n" + "=" * 70)
print("Ready to start recording!")
print("=" * 70)
print("\nControls:")
print("  Move leader arm to teleoperate follower arm")
print("  's' - Mark episode as success")
print("  'esc' - Mark episode as failure")
print("  'r' - Re-record current episode")
print("=" * 70)
print()

# Main recording loop
episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    # Main record loop
    record_loop(
        robot=follower,
        events=events,
        fps=FPS,
        teleop=leader,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=leader_joints_to_ee,
        robot_action_processor=ee_to_follower_joints,
        robot_observation_processor=follower_joints_to_ee,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (
        episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            teleop=leader,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=leader_joints_to_ee,
            robot_action_processor=ee_to_follower_joints,
            robot_observation_processor=follower_joints_to_ee,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    # Save episode
    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
leader.disconnect()
follower.disconnect()
listener.stop()

dataset.finalize()
dataset.push_to_hub()

print("\n" + "=" * 70)
print("✅ Recording complete!")
print("=" * 70)
print(f"Total episodes recorded: {episode_idx}")
print(f"Dataset: {HF_REPO_ID}")
print("\nNext steps:")
print("1. Review your recorded demonstrations")
print("2. Train ACT model:")
print(f"   lerobot-train \\")
print(f"     --dataset.repo_id={HF_REPO_ID} \\")
print(f"     --policy.type=act \\")
print(f"     --output_dir=outputs/cup_to_bottle_act")
print("=" * 70)
