from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt

print("📡 Connecting to CoppeliaSim (Trajectory Generator)...")
client = RemoteAPIClient()
sim = client.require('sim')

# ==========================================
# 1. LOAD CUP DATA
# ==========================================
data = np.load('cup_trajectory.npy', allow_pickle=True).item()
t_list  = np.array(data['t'])
cup_x, cup_y, cup_z = np.array(data['x']), np.array(data['y']), np.array(data['z'])
cup_r, cup_p = np.array(data['roll']), np.array(data['pitch'])

dt = 0.05
steps = len(t_list)

# ==========================================
# 2. FIND GRAB ZONE 
# ==========================================
dist = np.sqrt(cup_x**2 + cup_y**2)
grab_steps = np.where((dist >= 0.20) & (dist <= 0.72))[0]
step_A = grab_steps[0]                     
step_B = grab_steps[len(grab_steps)//2]    
step_C = grab_steps[-1] - 10               

# ==========================================
# 3. 🌟 PREP POSE (ท่าคว่ำมือลงพื้น 90 องศา)
# ==========================================
ef_handle = sim.getObjectHandle('/yaskawa/gripperEF')
joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]

# ท่าคว่ำมือ (เช็คให้ชัวร์ว่า -90 หรือ +90 หุ่นถึงจะคว่ำมือสวยๆ)
q_prep = np.array([0.0, np.radians(-30.0), np.radians(-60.0), 0.0, 0.0, 0.0])
for i, h in enumerate(joint_handles):
    sim.setJointPosition(h, float(q_prep[i]))
sim.step() 

pos_home  = sim.getObjectPosition(ef_handle, -1)
ori_home  = sim.getObjectOrientation(ef_handle, -1)

# ==========================================
# 4. 🌟 WAIT & SNAP GRASP (ท่าดักรอแล้วฉกเข้าข้างๆ!)
# ==========================================
cup_height = 0.12334
r_offset = 0.00001# ถอยแค่ระยะรัศมีแก้ว (3.5 ซม.) ผิวสัมผัสพอดีเป๊ะ

# 🌟 ฟังก์ชันพระเอกของเรา เอากลับมาแล้วครับ!
def quintic_segment(p_start, p_end, n_steps):
    if n_steps <= 0: return np.zeros((3, 0))
    t = np.linspace(0, 1, n_steps)
    s = 10*(t**3) - 15*(t**4) + 6*(t**5)
    path = np.zeros((3, n_steps))
    for i in range(3): path[i, :] = p_start[i] + (p_end[i] - p_start[i]) * s
    return path

# 1. ทิศทางพุ่งเข้าหาแก้ว
base_to_cup = np.array([cup_x[step_B], cup_y[step_B]])
radial_dir = base_to_cup / np.linalg.norm(base_to_cup)

# 2. 🎯 จุดตะปบ (จุดที่แก้วไหลมาอยู่ตรงหน้าพอดี)
z_tune = -0.02
p_grab = [
    cup_x[step_B] - (radial_dir[0] * r_offset), 
    cup_y[step_B] - (radial_dir[1] * r_offset), 
    cup_z[step_B] + (cup_height / 2) + z_tune  # จับกลางแก้ว
]

# 3. 🛡️ จุดดักรอ (ถอยออกมาด้านข้าง 10 ซม. ยืนรอนิ่งๆ ไม่ต้องวิ่งตาม)
p_wait = [
    p_grab[0] - (radial_dir[0] * 0.10), 
    p_grab[1] - (radial_dir[1] * 0.10), 
    p_grab[2]  # รอที่ความสูงเดียวกับตอนจะจับเลย จะได้ไม่ต้องขยับเยอะ
]

p_home = list(pos_home)
p_via  = [(p_home[0] + p_wait[0])/2, (p_home[1] + p_wait[1])/2, p_home[2] + 0.30]

# 4. 🌟 แบ่ง Step การเคลื่อนที่แบบง่ายๆ!
# สเต็ป 1: จาก Home ไป Via
seg_1a = quintic_segment(p_home, p_via, step_A // 2)
# สเต็ป 2: จาก Via ไปรอที่ p_wait
seg_1b = quintic_segment(p_via, p_wait, step_A - (step_A // 2))

# สเต็ป 3: จอดรอที่ p_wait นิ่งๆ จนกว่าแก้วจะไหลมาใกล้ถึง
wait_time = max((step_B - 15) - step_A, 2)  # รอจนถึงก่อนวินาทีจับนิดนึง
seg_wait = quintic_segment(p_wait, p_wait, wait_time)  

# สเต็ป 4: ⚡ แทงมือเข้าข้างๆ! (ใช้เวลาแค่ 15 Step พุ่ง 10 ซม.)
seg_snap = quintic_segment(p_wait, p_grab, 15)  


# สเต็ป 5: ยกขึ้นทันทีที่จับเสร็จ!
p_lift_end = [p_grab[0], p_grab[1], p_grab[2] + 0.15]
seg_lift = quintic_segment(p_grab, p_lift_end, step_C - step_B)  

# สเต็ป 6: ค้างไว้ข้างบน
seg_hold = quintic_segment(p_lift_end, p_lift_end, steps - step_C)  

# รวมร่างแกน X, Y, Z
ef_x = np.concatenate([seg_1a[0], seg_1b[0], seg_wait[0], seg_snap[0], seg_lift[0], seg_hold[0]])[:steps]
ef_y = np.concatenate([seg_1a[1], seg_1b[1], seg_wait[1], seg_snap[1], seg_lift[1], seg_hold[1]])[:steps]
ef_z = np.concatenate([seg_1a[2], seg_1b[2], seg_wait[2], seg_snap[2], seg_lift[2], seg_hold[2]])[:steps]

# ==========================================
# 5. 🌟 ORIENTATION: เอียงตามสายพาน + หันหน้าเข้าหาแก้ว
# ==========================================
R_conveyor = R.from_euler('xyz', [np.radians(-5.0), np.radians(10.0), 0.0])

# เอาเครื่องหมายลบออก! ให้แกน Z มือพุ่งชี้ไปหาแก้ว
z_target = np.array([radial_dir[0], radial_dir[1], 0.0]) 

x_up_world = np.array([0.0, 0.0, 1.0])
x_target = R_conveyor.apply(x_up_world) 

y_target = np.cross(z_target, x_target)
y_target /= np.linalg.norm(y_target)
z_target = np.cross(x_target, y_target)

R_target_mat = np.column_stack((x_target, y_target, z_target))
R_grab = R.from_matrix(R_target_mat)
R_home = R.from_euler('XYZ', ori_home)

slerp_func = Slerp([0, 1], R.concatenate([R_home, R_grab]))

ef_roll, ef_pitch, ef_yaw = np.zeros(steps), np.zeros(steps), np.zeros(steps)
for i in range(steps):
    t_norm = i / max(step_B, 1) if i < step_B else 1.0
    s = 10*(t_norm**3) - 15*(t_norm**4) + 6*(t_norm**5) if t_norm < 1.0 else 1.0
    rot = slerp_func([s])[0].as_euler('XYZ')
    ef_roll[i], ef_pitch[i], ef_yaw[i] = rot[0], rot[1], rot[2]

gripper = np.full(steps, 0.04)
# ให้มันวิ่งตีคู่ไปก่อน 20 step (1 วินาที) พอ Error เป็น 0 แน่ๆ ค่อยสับหนีบ!
grip_step = step_B + 20   
gripper[step_B : steps] = -0.40  


np.save('ef_trajectory.npy', {
    't': t_list[:steps].tolist(), 'x': ef_x.tolist(), 'y': ef_y.tolist(), 'z': ef_z.tolist(),
    'roll': ef_roll.tolist(), 'pitch': ef_pitch.tolist(), 'yaw': ef_yaw.tolist(),
    'gripper': gripper.tolist(), 'step_A': int(step_A), 'step_B': int(step_B), 'step_C': int(step_C)
})
print("✅ Saved Side-Grasp Trajectory!")

# ==========================================
# 6. 📊 PLOT TRAJECTORY (3D & 2D MULTI-VIEW)
# ==========================================
print("📈 Generating 3D and 2D Trajectory Plots...")

fig = plt.figure(figsize=(16, 10))

# ------------------------------------------
# 🖥️ จอ 1: ภาพรวม 3 มิติ (Top-Left)
# ------------------------------------------
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(cup_x, cup_y, cup_z, 'k--', alpha=0.5, label='Cup Path')
ax1.scatter(cup_x[step_B], cup_y[step_B], cup_z[step_B], color='red', s=50)
ax1.plot(ef_x, ef_y, ef_z, 'b-', linewidth=2, label='EE Trajectory')

ax1.scatter(*p_home, color='green', s=100, marker='o', label='1. Home')
ax1.scatter(*p_via, color='orange', s=80, marker='^', label='2. Via')
ax1.scatter(*p_wait, color='purple', s=80, marker='s', label='3. Wait')
ax1.scatter(*p_grab, color='red', s=150, marker='X', label='4. Snap/Grab') # แก้ตัวแปรเป็น p_grab
ax1.scatter(*p_lift_end, color='cyan', s=100, marker='*', label='5. Lift')

ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
ax1.set_title('3D Overall Trajectory', fontweight='bold')
ax1.legend(fontsize=8)

# ------------------------------------------
# 🖥️ จอ 2: ภาพมุมสูง X-Y Plane (Top-Right)
# เช็คว่ามือแทงเข้าข้างแก้วพอดีมั้ย หรือแทงทะลุ!
# ------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(cup_x, cup_y, 'k--', alpha=0.5, label='Cup')
ax2.plot(ef_x, ef_y, 'b-', linewidth=2, label='EE')
ax2.scatter(p_wait[0], p_wait[1], color='purple', s=100, marker='s', label='Wait Point')
ax2.scatter(p_grab[0], p_grab[1], color='red', s=150, marker='X', label='Snap Point')
ax2.set_xlabel('X Axis (m)'); ax2.set_ylabel('Y Axis (m)')
ax2.set_title('TOP VIEW (X-Y Plane)', fontweight='bold')
ax2.grid(True, linestyle=':'); ax2.axis('equal'); ax2.legend(fontsize=8)

# ------------------------------------------
# 🖥️ จอ 3: ภาพมุมหน้า X-Z Plane (Bottom-Left)
# เช็คว่ามือมุดดินทะลุสายพาน หรือลอยข้ามหัวแก้วมั้ย!
# ------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(cup_x, cup_z, 'k--', alpha=0.5, label='Cup Base')
ax3.plot(ef_x, ef_z, 'b-', linewidth=2, label='EE Height')
ax3.scatter(p_wait[0], p_wait[2], color='purple', s=100, marker='s')
ax3.scatter(p_grab[0], p_grab[2], color='red', s=150, marker='X')
ax3.set_xlabel('X Axis (m)'); ax3.set_ylabel('Z Axis (Height) (m)')
ax3.set_title('FRONT VIEW (X-Z Plane)', fontweight='bold')
ax3.grid(True, linestyle=':'); ax3.axis('equal')

# ------------------------------------------
# 🖥️ จอ 4: กราฟจังหวะเวลา Position vs Time (Bottom-Right)
# เช็คอาการ "ยืนรอ" (เส้นราบ) และ "พุ่งฉก" (เส้นชัน)
# ------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
time_arr = t_list[:steps]
ax4.plot(time_arr, ef_x, 'r-', linewidth=2, label='EE X')
ax4.plot(time_arr, ef_y, 'g-', linewidth=2, label='EE Y')
ax4.plot(time_arr, ef_z, 'b-', linewidth=2, label='EE Z')
ax4.axvline(x=time_arr[step_B], color='k', linestyle='--', linewidth=2, label='🔥 SNAP TIME!')

ax4.set_xlabel('Time (Seconds)')
ax4.set_ylabel('Position Value (m)')
ax4.set_title('TIMING: X, Y, Z vs Time', fontweight='bold')
ax4.grid(True, linestyle=':'); ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig('trajectory_dashboard.png', dpi=300)
print("✅ Dashboard saved successfully to -> trajectory_dashboard.png")
