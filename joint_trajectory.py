from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ==========================================
# CONNECT & SETUP
# ==========================================
client = RemoteAPIClient()
sim = client.require('sim')

joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
ef_handle   = sim.getObjectHandle('/yaskawa/gripperEF')
mico_motor1 = sim.getObjectHandle('/yaskawa/MicoHand/fingers12_motor1')
mico_motor2 = sim.getObjectHandle('/yaskawa/MicoHand/fingers12_motor2')

q_min = np.array([-2.9670, -1.7453, -3.1415, -3.4906, -2.0943, -6.2831])
q_max = np.array([ 2.9670,  2.6179,  1.2217,  3.4906,  2.0943,  6.2831])

# ==========================================
# MATH & IK FUNCTIONS
# ==========================================
def get_T(handle, ref=-1):
    m = sim.getObjectMatrix(handle, ref)
    return np.array([
        [m[0], m[1], m[2], m[3]],
        [m[4], m[5], m[6], m[7]],
        [m[8], m[9], m[10], m[11]],
        [0,    0,    0,    1]
    ])

def get_jacobian_geometric(joint_handles, ef_handle):
    T_ef = get_T(ef_handle)
    p_e  = T_ef[:3, 3]
    J    = np.zeros((6, 6))
    for i, h in enumerate(joint_handles):
        T_i      = get_T(h)
        z_i      = T_i[:3, 2]
        p_i      = T_i[:3, 3]
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J

def get_adaptive_lambda(J):
    w = np.sqrt(abs(np.linalg.det(J @ J.T)))
    w0, lambda_max = 0.04, 0.4 
    if w < w0: return lambda_max * (1.0 - (w / w0)**2) + 0.05 
    else: return 0.05  

# ==========================================
# LOAD TRAJECTORY
# ==========================================
print("📂 Loading perfectly smooth trajectory...")
ef_data = np.load('ef_trajectory.npy', allow_pickle=True).item()

ef_x, ef_y, ef_z = np.array(ef_data['x']), np.array(ef_data['y']), np.array(ef_data['z'])
ef_roll, ef_pitch, ef_yaw = np.array(ef_data['roll']), np.array(ef_data['pitch']), np.array(ef_data['yaw'])
gripper_cmd = np.array(ef_data['gripper'])

step_A, step_B, step_C = ef_data['step_A'], ef_data['step_B'], ef_data['step_C']
steps, dt = len(ef_x), 0.05

# 🌟 เตรียมตัวแปรสำหรับเก็บข้อมูลไปพล็อตทราฟ
actual_x, actual_y, actual_z = np.zeros(steps), np.zeros(steps), np.zeros(steps)
pos_err_list = np.zeros(steps)
ori_err_list = np.zeros(steps)
time_axis = np.arange(steps) * dt

# ==========================================
# 🌟 START SIMULATION & INSTANT PREP
# ==========================================
sim.setStepping(True)
sim.startSimulation()

print("\n🚀 Waking up robot exactly at t=0s...")
q_prep = np.array([0.0, np.radians(-30.0), np.radians(-60.0), 0.0, 0.0, 0.0])

for i, h in enumerate(joint_handles):
    sim.setJointPosition(h, float(q_prep[i]))       
    sim.setJointTargetPosition(h, float(q_prep[i])) 

sim.setJointForce(mico_motor1, 50.0)
sim.setJointForce(mico_motor2, 50.0)
sim.setJointTargetVelocity(mico_motor1, 0.15)
sim.setJointTargetVelocity(mico_motor2, 0.15)

# 🌟 สร้างตัวแปรบอกว่าเราผลาญเวลาไปกี่เฟรม
sync_offset = 3 
for _ in range(sync_offset): 
    sim.step()

for _ in range(3): sim.step()

q = np.array([sim.getJointPosition(h) for h in joint_handles])
print(f"✅ Ready! Prep Pose = {np.round(np.degrees(q), 1)} deg. Switching to IK...")

# ==========================================
# MAIN IK LOOP
# ==========================================
j_traj = np.zeros((6, steps))
j_traj[:,0] = q
gripped = False

for step in range(sync_offset, steps):
    pos_cur = sim.getObjectPosition(ef_handle, -1)
    ori_cur = sim.getObjectOrientation(ef_handle, -1)
    
    # เก็บข้อมูลระยะจริงเพื่อนำไปพล็อตกราฟ
    actual_x[step] = pos_cur[0]
    actual_y[step] = pos_cur[1]
    actual_z[step] = pos_cur[2]
    
    # 1. Orientation Error 
    R_tar = R.from_euler('XYZ', [ef_roll[step], ef_pitch[step], ef_yaw[step]])
    R_cur = R.from_euler('XYZ', ori_cur)
    w_err = (R_tar * R_cur.inv()).as_rotvec() 

    # 2. Position Error
    p_err = np.array([ef_x[step] - pos_cur[0], ef_y[step] - pos_cur[1], ef_z[step] - pos_cur[2]])
    
    # เก็บค่า Error ลง List
    pos_err_list[step] = np.linalg.norm(p_err) * 1000
    ori_err_list[step] = np.linalg.norm(w_err)

    # 3. Feedforward 
    V_ff = np.zeros(3) if step == 0 else np.array([(ef_x[step]-ef_x[step-1])/dt, (ef_y[step]-ef_y[step-1])/dt, (ef_z[step]-ef_z[step-1])/dt])

    # 4. Combine Errors
    Kp_pos, Kp_ori = 5.0, 2.0  
    X_dot = np.zeros(6)
    X_dot[:3] = V_ff + (p_err * Kp_pos) 
    X_dot[3:] = w_err * Kp_ori 

    # Limit speed to prevent jerks
    v_norm, w_norm = np.linalg.norm(X_dot[:3]), np.linalg.norm(X_dot[3:])
    if v_norm > 2.0: X_dot[:3] = (X_dot[:3] / v_norm) * 0.8
    if w_norm > 1.5: X_dot[3:] = (X_dot[3:] / w_norm) * 1.5

    # 5. Jacobian & DLS
    J = get_jacobian_geometric(joint_handles, ef_handle)
    lambda_damp = get_adaptive_lambda(J)
    JJT   = J @ J.T
    J_dls = J.T @ np.linalg.inv(JJT + lambda_damp**2 * np.eye(6))
    q_dot = J_dls @ X_dot

    # ดักคอ Joint 2 (ไหล่) ไม่ให้เหวี่ยงแรงเกินไป
    q_dot[1] = np.clip(q_dot[1], -1.5, 1.5) 

    q_dot = np.clip(q_dot, -2.5, 2.5)
    q = q + q_dot * dt
    q = np.clip(q, q_min, q_max)

    for i, h in enumerate(joint_handles):
        sim.setJointPosition(h, float(q[i]))

    # 6. Gripper
    current_grip_vel = float(gripper_cmd[step])
    sim.setJointForce(mico_motor1, 50.0)
    sim.setJointForce(mico_motor2, 50.0)
    sim.setJointTargetVelocity(mico_motor1, current_grip_vel)
    sim.setJointTargetVelocity(mico_motor2, current_grip_vel)

    if current_grip_vel < 0 and not gripped:
        print(f"\n🔒 GRIPPING at step {step} (t={step*dt:.2f}s)! Watch the error drop!")
        gripped = True

    j_traj[:, step] = q

    if step % 50 == 0:
        print(f"Tracking: Step {step:03d}/{steps} | Pos Err = {pos_err_list[step]:>5.1f} mm | Ori Err = {ori_err_list[step]:.4f} rad")

    sim.step()

sim.stopSimulation()
print("\n🎉 MISSION ACCOMPLISHED! THE PERFECT POUR!")

print("\n🔍 CHECKING JOINT LIMITS AT FINAL STEP:")
for i in range(6):
    print(f"J{i+1}: {np.degrees(q[i]):>6.1f} deg | Limit: [{np.degrees(q_min[i]):>6.1f}, {np.degrees(q_max[i]):>6.1f}]")

# ==========================================
# 📊 GENERATE PERFORMANCE PLOTS
# ==========================================
print("\n📈 Generating IK Performance Plots...")
fig = plt.figure(figsize=(14, 10))

# 1. กราฟ Position Error (ความแม่นยำในการวิ่งตาม)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(time_axis, pos_err_list, 'r-', linewidth=1.5)
ax1.axvline(x=step_B*dt, color='k', linestyle='--', alpha=0.5, label='Grab Moment')
ax1.set_title('Position Tracking Error (mm)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Error (mm)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# 2. กราฟ Orientation Error (ความแม่นยำในการเอียงข้อมือ)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(time_axis, ori_err_list, 'b-', linewidth=1.5)
ax2.axvline(x=step_B*dt, color='k', linestyle='--', alpha=0.5, label='Grab Moment')
ax2.set_title('Orientation Tracking Error (rad)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (rad)')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

# 3. กราฟ 3D เทียบเส้นทางเป้าหมาย vs วิ่งจริง
ax3 = fig.add_subplot(2, 2, (3, 4), projection='3d')
ax3.plot(ef_x, ef_y, ef_z, 'k--', linewidth=2, alpha=0.7, label='Target Trajectory')
ax3.plot(actual_x[1:], actual_y[1:], actual_z[1:], 'g-', linewidth=2, label='Actual Robot Path')
ax3.scatter(actual_x[step_B], actual_y[step_B], actual_z[step_B], color='red', s=100, marker='X', label='Grab Point')
ax3.set_title('3D Path Tracking Performance', fontsize=12, fontweight='bold')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_zlabel('Z (m)')
ax3.legend()

plt.tight_layout()
plt.savefig('ik_performance_plot.png', dpi=300)
print("✅ Plot saved successfully to -> ik_performance_plot.png")
