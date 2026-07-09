from rosbags.highlevel import AnyReader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from mpx.utils.kf_utils import quat_to_rot

class LegKILOReader():
    """Class to read ROS .bag file and ground truth .txt-file from Leg KILO dataset (https://github.com/ouguangjun/legkilo-dataset).

    The .bag file contains the following topics:
    - /high_state: state of Unitree-SDK
    - /state_SDK: odometry estimation from SDK/robot
    - /imu_raw: raw IMU measurements
    - /points_raw: raw LiDAR point-cloud scans

    There are 3 frames:
        _b - base frame of the robot (given in .bag file)
        _o - odom frame / world frame based on unitree initialization (given in .bag file, manual transformation)
        _w - world/map frame defined by LiDaR-map (given in .txt file)
    
    The ground truth data is used in the beginning to allign the orientation of the odom frame with the world frame.
    """

    def __init__(self, rosbag_path, gt_path, output_dir, file_name, save_rosbag=True, read_again=False):
        self.rosbag_path = Path(rosbag_path)
        self.gt_path = gt_path
        self.output_dir = output_dir
        self.file_name = file_name

        # read gt data
        print(f"Reading ground truth from {self.gt_path}")
        self.read_gt_file()

        # read rosbag if not already saved
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.isfile(f"{self.output_dir}/{self.file_name}.npz") or read_again:
            print(f"Reading rosbag data from {self.rosbag_path}")
            self.legkilo_data = self.read_rosbag_file(save=save_rosbag)
        else:
            print(f"Loading data from {self.output_dir}/{self.file_name}.npz")
            self.legkilo_data = np.load(f"{self.output_dir}/{self.file_name}.npz")
        
        self.num_points = len(self.legkilo_data["time_state"])
    
    def inspect_rosbag_file(self, check_topic_eq=False):
        """Print example message for each topic

        Args:
            check_topic_eq (bool, optional): Check if values in topics are equal. Defaults to False.
        """
        def dump(obj, prefix=""):
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    dump(v, f"{prefix}{k}.")
            else:
                print(prefix[:-1], "=", obj)
        
        def compare_dicts(d1, d2, name1="d1", name2="d2", atol=1e-6):
            common_keys = set(d1.keys()) & set(d2.keys())
            for key in common_keys:
                arr1 = np.array(d1[key])
                arr2 = np.array(d2[key])
                if arr1.shape != arr2.shape:
                    print(f"[{key}] shape mismatch: {name1}={arr1.shape} vs {name2}={arr2.shape}")
                    continue
                equal = np.allclose(arr1, arr2, atol=atol)
                print(f"[{key}] {name1} vs {name2}: {'equal' if equal else 'not equal'}")
                if not equal:
                    diff = np.abs(arr1 - arr2)
                    print(f"    max diff: {diff.max()}")

        with AnyReader([self.rosbag_path]) as reader:
            print(30*"=", "Topics", 30*"=")
            topics_of_interest = []

            # read topics from .bag file
            for c in reader.connections:
                print(f"{c.topic:40s}  {c.msgtype}")
                topics_of_interest.append(c.topic)

            if check_topic_eq:
                high_state = {}
                state_sdk = {}
                imu_raw = {}
            for topic_of_interest in topics_of_interest:
                conns = [c for c in reader.connections if c.topic == topic_of_interest]
                if not conns:
                    raise RuntimeError(f"Topic not found: {topic_of_interest}")

                info_print = True
                for conn, t, raw in reader.messages(connections=conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    if info_print:
                        print(f"\n{20*"="} Example for {conn.topic} ({conn.msgtype}) {20*"="}")
                        dump(msg)
                        info_print = False

                    if check_topic_eq:
                        if conn.topic == "/high_state":
                            high_state.setdefault("orient_quat", []).append(msg.imu.quaternion)
                            high_state.setdefault("base_acc", []).append(msg.imu.accelerometer)
                            high_state.setdefault("base_pos", []).append(msg.position)
                            high_state.setdefault("base_lin_vel", []).append(msg.velocity)
                            high_state.setdefault("base_ang_vel", []).append(msg.imu.gyroscope)
                        if conn.topic == "/state_SDK":
                            state_sdk.setdefault("orient_quat", []).append([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
                            state_sdk.setdefault("base_pos", []).append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                            state_sdk.setdefault("base_lin_vel", []).append([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                            state_sdk.setdefault("base_ang_vel", []).append([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
                        if conn.topic == "/imu_raw":
                            imu_raw.setdefault("orient_quat", []).append([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
                            imu_raw.setdefault("base_ang_vel", []).append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                            imu_raw.setdefault("base_acc", []).append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
                    else:
                        break
            
            if check_topic_eq:
                print("\nCheck if fields with same name are equal between messages:")
                compare_dicts(high_state, state_sdk, "high_state", "state_sdk")
                compare_dicts(high_state, imu_raw, "high_state", "imu_raw")
                compare_dicts(state_sdk, imu_raw, "state_sdk", "imu_raw")
    
    def read_rosbag_file(self, save=True):        
        data = {
            "base_orient_quat": [],
            "base_orient_rpy": [],
            "base_ang_vel": [],
            "base_acc": [],

            # base vel estimation of unitree sdk
            "base_vel_b": [],
            "base_vel_o": [],
            "base_vel_w": [],

            # base pos estimation of unitree sdk
            "base_pos_b": [],
            "base_pos_o": [],
            "base_pos_o2": [],
            "base_pos_w": [],

            "joint_pos": [],
            "joint_vel": [],
            "joint_acc": [],
            "joint_torque": [],
        
            "foot_force": [],
            "contact_pos": [],
            "time_state": [],
            "dt": [],
        }

        with AnyReader([self.rosbag_path]) as reader:
            for connection, timestamp, rawdata in tqdm(
                reader.messages(),
                desc="Reading ROS .bag file",
                total=670538,
            ):
                if connection.topic != "/high_state":
                    continue

                msg = reader.deserialize(rawdata, connection.msgtype)

                # orientation between base frame and initial odom frame
                quat = msg.imu.quaternion  # [x, y, z, w]
                base_orient = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)  # [w, x, y, z]
                data["base_orient_quat"].append(base_orient)
                data["base_orient_rpy"].append(np.array(msg.imu.rpy, dtype=float))

                # base frame
                data["base_ang_vel"].append(np.array(msg.imu.gyroscope, dtype=float))
                data["base_acc"].append(np.array(msg.imu.accelerometer, dtype=float))
                vel_b = np.array(msg.velocity, dtype=float)
                data["base_vel_b"].append(vel_b)

                # odom frame
                data["base_pos_o"].append(np.array(msg.position, dtype=float))
                vel_o = quat_to_rot(base_orient) @ vel_b
                data["base_vel_o"].append(np.array(vel_o, dtype=float))

                # frame-less
                data["joint_pos"].append([msg.motorState[i].q for i in range(12)])
                data["joint_vel"].append([msg.motorState[i].dq for i in range(12)])
                data["joint_acc"].append([msg.motorState[i].ddq for i in range(12)])
                data["joint_torque"].append([msg.motorState[i].tauEst for i in range(12)])
                data["contact_pos"].append([[foot.x, foot.y, foot.z] for foot in msg.footPosition2Body])
                data["foot_force"].append(np.array(msg.footForce, dtype=float))

                # time
                t = msg.stamp.sec + msg.stamp.nanosec * 1e-9
                data["time_state"].append(t)

        for key in data:
            data[key] = np.array(data[key])

        # integrate base_vel_o to get odom frame representation of base_pos. sanity check of position in /high_state
        pos_o = np.zeros((len(data["time_state"]), 3), dtype=float)
        pos_o[0] = data["base_pos_o"][0]
        times = data["time_state"]
        vels = data["base_vel_o"]

        dts = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            dts.append(dt)
            pos_o[i] = pos_o[i - 1] + vels[i - 1] * dt
        
        data["base_pos_o2"] = pos_o
        data["dt"] = np.array(dts)

        print("Converting base pos from odom frame to world frame.")
        data["base_pos_w"] = self.convert_odom_to_map(odom_data=data["base_pos_o"],
                                                      first_quat=data["base_orient_quat"][0],
                                                      with_offset=True)

        print("Converting base vel from odom frame to world frame.")
        data["base_vel_w"] = self.convert_odom_to_map(odom_data=data["base_vel_o"],
                                                      first_quat=data["base_orient_quat"][0],
                                                      with_offset=False)

        if save:
            np.savez(f"{self.output_dir}/{self.file_name}.npz", **data)
            print(f"Saved data in: {self.output_dir}")

        return data
    
    def convert_odom_to_map(self, odom_data, first_quat, with_offset=True):
        """Convert base position and velocity from odom frame to world/map frame to allign rosbag orientation with ground truth.

        Using the first rotation matrix from ground truth data (world/map frame) and first 
        rotation matrix from rosbag file (odom frame) to create a rotation between those frames.

        Args:
            odom_data (np.ndarray): Base pos or vel in odom frame from rosbag file
            with_offset (bool, optional): To correct offset shift of base pos. Defaults to True.

        Returns:
            np.ndarray: Converted datapoins in world/map frame
        """

        R_o = quat_to_rot(first_quat) # first rotation matrix from rosbag (odom frame)

        # convert order of gt quaternion
        gt_quat = self.gt_quaternions[0]
        gt_orient = np.array([gt_quat[3], gt_quat[0], gt_quat[1], gt_quat[2]], dtype=float)
        R_gt = quat_to_rot(gt_orient) # first rotation matrix from gt (world/map frame)

        R_w = R_gt @ R_o.T # rotation matrix between odom and world/map frame

        # using first gt pos and rosbag pos to compute the offset
        offset = self.gt_positions[0] - R_w @ odom_data[0] if with_offset else np.zeros(3)

        # converting base pos in odom frame to world/map frame: rotation + translation
        p_world = (R_w @ odom_data.T).T + offset

        return p_world
    
    def read_gt_file(self):
        gt_data = np.loadtxt(Path(self.gt_path), comments='#')

        self.gt_timestamps = gt_data[:, 0]
        self.gt_positions = gt_data[:, 1:4]   # x, y, z
        self.gt_quaternions = gt_data[:, 4:8]  # qx, qy, qz, qw
