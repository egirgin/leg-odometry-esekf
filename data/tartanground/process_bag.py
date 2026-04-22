import csv
from pathlib import Path
from rosbags.highlevel import AnyReader

root_path = Path("anymal")
output_root = Path("processed")

if not root_path.exists():
    print(f"[FATAL] Root directory not found: {root_path}")
    exit(1)

header = ['sec', 'nanosec']
for i in range(4): header.append(f'foot_force_{i}')
for i in range(12): header.extend([f'motor_{i}_q', f'motor_{i}_dq', f'motor_{i}_ddq', f'motor_{i}_tau_est'])
header.extend([
    'p_x', 'p_y', 'p_z',
    'vel_lin_x', 'vel_lin_y', 'vel_lin_z',
    'vel_ang_x', 'vel_ang_y', 'vel_ang_z'
])

for traj_dir in root_path.glob("*/Data_anymal/*"):
    if not traj_dir.is_dir():
        continue
        
    rosbags_dir = traj_dir / "rosbags"
    if not rosbags_dir.exists():
        continue
        
    bag_files = list(rosbags_dir.glob("*.bag"))
    if not bag_files:
        continue

    env_name = traj_dir.parents[1].name
    traj_name = traj_dir.name
    
    out_dir = output_root / env_name / traj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_bag in bag_files:
        output_csv = out_dir / f"{target_bag.stem}_bag.csv"
        print(f"[ACTION] -> Extracting {target_bag.name} to {output_csv}")

        try:
            with AnyReader([target_bag]) as reader:
                conn = next((c for c in reader.connections if c.topic == '/state_estimator/anymal_state'), None)
                
                if not conn:
                    print(f"[ERROR] /state_estimator/anymal_state topic not found in {target_bag.name}. Skipping.")
                    continue

                with open(output_csv, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)

                    t0_sec, t0_nano = None, None
                    p0_x = p0_y = p0_z = None
                    row_count, neg_time_errors = 0, 0

                    for _, timestamp, rawdata in reader.messages(connections=[conn]):
                        msg = reader.deserialize(rawdata, conn.msgtype)
                        curr_sec, curr_nano = msg.header.stamp.sec, msg.header.stamp.nanosec
                        
                        curr_px = msg.pose.pose.position.x
                        curr_py = msg.pose.pose.position.y
                        curr_pz = msg.pose.pose.position.z

                        if t0_sec is None:
                            t0_sec, t0_nano = curr_sec, curr_nano
                            p0_x, p0_y, p0_z = curr_px, curr_py, curr_pz
                        
                        rel_sec, rel_nano = curr_sec - t0_sec, curr_nano - t0_nano
                        
                        if rel_nano < 0:
                            rel_sec -= 1
                            rel_nano += int(1e9)
                            
                        if row_count == 0 and (rel_sec != 0 or rel_nano != 0):
                            print(f"[INVALID] Zero alignment failed. Read: {rel_sec}s {rel_nano}ns")
                        if rel_sec < 0:
                            neg_time_errors += 1
                            
                        row = [rel_sec, rel_nano]
                        
                        for i in range(4): row.append(msg.contacts[i].wrench.force.z)
                        for i in range(12):
                            row.append(msg.joints.position[i])
                            row.append(msg.joints.velocity[i])
                            row.append(msg.joints.acceleration[i])
                            row.append(msg.joints.effort[i])
                            
                        row.extend([
                            curr_px - p0_x, curr_py - p0_y, curr_pz - p0_z,
                            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                            msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z
                        ])
                        
                        writer.writerow(row)
                        row_count += 1

            if neg_time_errors > 0:
                print(f"[INVALID] {neg_time_errors} rows contained negative timestamps in {target_bag.name}.")
                
        except Exception as e:
            print(f"[FATAL] Failed to process {target_bag.name}: {str(e)}")