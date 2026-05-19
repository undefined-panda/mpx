from rosbags.highlevel import AnyReader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

class LegKILOReader():
    """Class to read ROS .bag file and ground truth .txt-file from Leg KILO dataset (https://github.com/ouguangjun/legkilo-dataset).

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
    
    def inspect_rosbag_file(self):
        def dump(obj, prefix=""):
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    dump(v, f"{prefix}{k}.")
            else:
                print(prefix[:-1], "=", obj)

        with AnyReader([self.rosbag_path]) as reader:
            print(30*"=", "Topics", 30*"=")
            topics_of_interest = []

            # read topics from .bag file
            for c in reader.connections:
                print(f"{c.topic:40s}  {c.msgtype}")
                topics_of_interest.append(c.topic)
            
            for topic_of_interest in topics_of_interest:
                conns = [c for c in reader.connections if c.topic == topic_of_interest]
                if not conns:
                    raise RuntimeError(f"Topic not found: {topic_of_interest}")

                for conn, t, raw in reader.messages(connections=conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    print(f"\n{20*"="} Example for {conn.topic} ({conn.msgtype}) {20*"="}")
                    dump(msg)
                    if conn.topic == "/high_state":
                        print("000000000")
                        footForce = msg.footForce
                        break
                break
    
    def read_rosbag_file(self, save=True):
        """_summary_
        Returns:
            _type_: _description_
        """
        
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

                # convert quaternion order to match conversion function
                quat = msg.imu.quaternion  # [x, y, z, w]
                base_orient = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)  # [w, x, y, z]

                data["base_orient_quat"].append(base_orient)
                data["base_orient_rpy"].append(np.array(msg.imu.rpy, dtype=float))
                data["base_ang_vel"].append(np.array(msg.imu.gyroscope, dtype=float))
                data["base_acc"].append(np.array(msg.imu.accelerometer, dtype=float))

                # base frame
                vel_b = np.array(msg.velocity, dtype=float)
                pos_b = np.array(msg.position, dtype=float)
                data["base_vel_b"].append(vel_b)
                data["base_pos_b"].append(pos_b)

                # odom frame
                R = quat_to_rot(base_orient)
                vel_o = R @ vel_b
                data["base_vel_o"].append(np.array(vel_o, dtype=float))

                data["joint_pos"].append([msg.motorState[i].q for i in range(12)])
                data["joint_vel"].append([msg.motorState[i].dq for i in range(12)])
                data["joint_acc"].append([msg.motorState[i].ddq for i in range(12)])
                data["joint_torque"].append([msg.motorState[i].tauEst for i in range(12)])

                data["contact_pos"].append([[foot.x, foot.y, foot.z] for foot in msg.footPosition2Body])
                data["foot_force"].append(np.array(msg.footForce, dtype=float))

                t = msg.stamp.sec + msg.stamp.nanosec * 1e-9
                data["time_state"].append(t)

        for key in data:
            data[key] = np.array(data[key])

        # integrate base_vel_o to get odom frame representation of base_pos
        pos_o = np.zeros((len(data["time_state"]), 3), dtype=float)
        times = data["time_state"]
        vels = data["base_vel_o"]

        dts = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            dts.append(dt)
            pos_o[i] = pos_o[i - 1] + vels[i - 1] * dt
        
        data["base_pos_o"] = pos_o
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

def quat_to_rot(orient) -> np.ndarray:
    """Convert quaternion to rotation matrix (source: https://cookierobotics.com/080/).

    Args:
        orient (np.ndarray | list): quaternion in [w, x, y, z] format

    Returns:
        np.ndarray: corresponding rotation matrix
    """

    w, x, y, z = orient
    R = np.array([
        [2*(w**2 + x**2) - 1, 2*(x*y - w*z)      , 2*(w*y + x*z)      ],
        [2*(x*y + w*z)      , 2*(w**2 + y**2) - 1, 2*(y*z - w*x)      ],
        [2*(x*z - w*y)      , 2*(y*z + w*x)      , 2*(w**2 + z**2) - 1]
    ])
    
    return R
