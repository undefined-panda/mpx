from rosbags.highlevel import AnyReader
from tqdm import tqdm
import numpy as np

def dump(obj, prefix=""):
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            dump(v, f"{prefix}{k}.")
    else:
        print(prefix[:-1], "=", obj)

def inspect_rosbag_file(bagpath):
    with AnyReader([bagpath]) as reader:
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

def read_rosbag_file(bagpath,
                     outpur_dir,
                     file_name,
                     order=[0,1,2, 3,4,5, 6,7,8, 9,10,11],
                     save=True):
    
    leg_kilo_base_orient = []
    leg_kilo_rpy = []
    leg_kilo_base_ang_vel = []
    leg_kilo_base_acc = []
    leg_kilo_base_vel = []
    leg_kilo_base_pos = []
    leg_kilo_joint_pos = []
    leg_kilo_joint_vel = []
    leg_kilo_joint_torque = []
    leg_kilo_foot_force = []
    leg_kilo_contact_pos = []
    leg_kilo_time_state = []

    with AnyReader([bagpath]) as reader:
        for connection, timestamp, rawdata in tqdm(reader.messages(), desc="Reading ROS .bag file", total=670538):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == '/high_state':
                quat = msg.imu.quaternion  # [x, y, z, w]
                temp_base_orient = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float) # [w, x, y, z]
                leg_kilo_base_orient.append(temp_base_orient) # base orient

                leg_kilo_rpy.append(msg.imu.rpy)

                leg_kilo_base_ang_vel.append(np.array(msg.imu.gyroscope, dtype=float)) # base ang vel
                leg_kilo_base_acc.append(np.array(msg.imu.accelerometer, dtype=float)) # base acc
                leg_kilo_base_vel.append(np.array(msg.velocity, dtype=float)) # base vel
                leg_kilo_base_pos.append(np.array(msg.position, dtype=float)) # base pos

                leg_kilo_joint_pos.append([msg.motorState[i].q for i in order])
                leg_kilo_joint_vel.append([msg.motorState[i].dq for i in order])
                leg_kilo_joint_torque.append([msg.motorState[i].tauEst for i in order])
                
                leg_kilo_contact_pos.append([[foot.x, foot.y, foot.z] for foot in msg.footPosition2Body]) # contact pos
                leg_kilo_foot_force.append(np.array(msg.footForce, dtype=float)) # foot force 

                leg_kilo_time_state.append(msg.stamp.sec + msg.stamp.nanosec * 1e-9)

    # transform to np.array
    leg_kilo_base_orient = np.array(leg_kilo_base_orient)
    leg_kilo_rpy = np.array(leg_kilo_rpy)
    leg_kilo_base_ang_vel = np.array(leg_kilo_base_ang_vel)
    leg_kilo_base_acc = np.array(leg_kilo_base_acc)
    leg_kilo_base_vel = np.array(leg_kilo_base_vel)
    leg_kilo_base_pos = np.array(leg_kilo_base_pos)
    leg_kilo_joint_pos = np.array(leg_kilo_joint_pos)
    leg_kilo_joint_vel = np.array(leg_kilo_joint_vel)
    leg_kilo_joint_torque = np.array(leg_kilo_joint_torque)
    leg_kilo_foot_force = np.array(leg_kilo_foot_force)
    leg_kilo_contact_pos = np.array(leg_kilo_contact_pos)
    leg_kilo_time_state = np.array(leg_kilo_time_state)

    if save:
        outpur_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            f"{outpur_dir}/{file_name}.npz",
            base_orient=leg_kilo_base_orient,
            rpy=leg_kilo_rpy,
            base_ang_vel=leg_kilo_base_ang_vel,
            base_acc=leg_kilo_base_acc,
            base_vel=leg_kilo_base_vel,
            base_pos=leg_kilo_base_pos,
            joint_pos=leg_kilo_joint_pos,
            joint_vel=leg_kilo_joint_vel,
            joint_torque=leg_kilo_joint_torque,
            foot_force=leg_kilo_foot_force,
            contact_pos=leg_kilo_contact_pos,
            time_state=leg_kilo_time_state,
        )

def compute_mean_dt(time_states):
    
    dts = []
    prev_t = None
    for i in range(len(time_states)):
        t = time_states[i]
        if prev_t is not None:
            dts.append(t - prev_t)
        prev_t = t

    dt = np.round(np.mean(dts), 4)

    return dt
