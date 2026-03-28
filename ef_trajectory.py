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
# 4. 🌟 SIDE-GRASP TRAJECTORY (หนีบข้างแก้วตามสั่ง!)
# ==========================================
cup_height = 0.12334
track_len = 5 

# พิกัดเป้าหมาย (กลางแก้ว)
p_grab_start = [cup_x[step_B], cup_y[step_B], cup_z[step_B] + (cup_height/2)]

# จุดรอ (ถอยออกมาจากข้างแก้ว 15 เซนติเมตร)
base_to_cup = np.array([p_grab_start[0], p_grab_start[1]])
radial_dir = base_to_cup / np.linalg.norm(base_to_cup)
p_wait = [
    p_grab_start[0] + radial_dir[0] * 0.15, 
    p_grab_start[1] + radial_dir[1] * 0.15, 
    p_grab_start[2] +0.05
]

p_home = list(pos_home)
p_via  = [(p_home[0] + p_wait[0])/2, (p_home[1] + p_wait[1])/2, p_home[2] + 0.30]

# จุดยก (ยกขึ้นตรงๆ หลังหนีบเสร็จ)
step_lift = step_B + track_len
p_lift_start = [cup_x[step_lift], cup_y[step_lift], cup_z[step_lift] + (cup_height/2)]
p_lift_end   = [p_lift_start[0], p_lift_start[1], p_lift_start[2] + 0.15]

def quintic_segment(p_start, p_end, n_steps):
    if n_steps <= 0: return np.zeros((3, 0))
    t = np.linspace(0, 1, n_steps)
    s = 10*(t**3) - 15*(t**4) + 6*(t**5)
    path = np.zeros((3, n_steps))
    for i in range(3): path[i, :] = p_start[i] + (p_end[i] - p_start[i]) * s
    return path

seg_1a = quintic_segment(p_home, p_via, step_A // 2)
seg_1b = quintic_segment(p_via, p_wait, step_A - (step_A // 2))
seg_2  = quintic_segment(p_wait, p_wait, max((step_B - 10) - step_A, 2))  
seg_3  = quintic_segment(p_wait, p_grab_start, 10)  

# วิ่งตีคู่ข้างแก้ว 5 Step
seg_track = np.zeros((3, track_len))
for i in range(track_len):
    seg_track[1, i] = cup_y[step_B + i]
    seg_track[2, i] = cup_z[step_B + i] + (cup_height/2)

seg_4  = quintic_segment(p_lift_start, p_lift_end, step_C - step_lift)  
seg_5  = quintic_segment(p_lift_end, p_lift_end, steps - step_C)  

ef_x = np.concatenate([seg_1a[0], seg_1b[0], seg_2[0], seg_3[0], seg_track[0], seg_4[0], seg_5[0]])[:steps]
ef_y = np.concatenate([seg_1a[1], seg_1b[1], seg_2[1], seg_3[1], seg_track[1], seg_4[1], seg_5[1]])[:steps]
ef_z = np.concatenate([seg_1a[2], seg_1b[2], seg_2[2], seg_3[2], seg_track[2], seg_4[2], seg_5[2]])[:steps]

# ==========================================
# 5. 🌟 ORIENTATION: เอียงตามสายพาน + หันหน้าเข้าหาแก้ว
# ==========================================
# 1. สร้าง Matrix ความเอียงของสายพาน (จากรูปที่ลูกพี่ส่งมา)
R_conveyor = R.from_euler('xyz', [np.radians(-5.0), np.radians(10.0), 0.0])

# 2. แกน Z ของมือ (Approach Vector) ต้องพุ่งเข้าหาแก้วในแนวระนาบ
z_target = np.array([-radial_dir[0], -radial_dir[1], 0.0])

# 3. แกน X ของมือ (Orientation Vector) แทนที่จะชี้ขึ้นฟ้าตรงๆ 
# ให้มัน "เอียงตามสายพาน" แทนครับ!
x_up_world = np.array([0.0, 0.0, 1.0])
x_target = R_conveyor.apply(x_up_world) # บิดแกน X ให้เอียงตามพื้นสายพาน

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
gripper[step_B : steps] = -0.40  

# ==========================================
# 🌟 ขอดักแก้บั๊กแกน X ตรง seg_track ให้ลูกพี่นิดนึงนะครับ (ใส่ทับของเดิมได้เลย)
# ==========================================
for i in range(track_len):
    seg_track[0, i] = p_grab_start[0]  # <--- เติมแกน X ให้มันล็อคระยะข้างแก้วไว้
    seg_track[1, i] = cup_y[step_B + i]
    seg_track[2, i] = cup_z[step_B + i] + (cup_height/2)

np.save('ef_trajectory.npy', {
    't': t_list[:steps].tolist(), 'x': ef_x.tolist(), 'y': ef_y.tolist(), 'z': ef_z.tolist(),
    'roll': ef_roll.tolist(), 'pitch': ef_pitch.tolist(), 'yaw': ef_yaw.tolist(),
    'gripper': gripper.tolist(), 'step_A': int(step_A), 'step_B': int(step_B), 'step_C': int(step_C)
})
print("✅ Saved Side-Grasp Trajectory!")

# ==========================================
# 6. 📊 PLOT TRAJECTORY (สร้างกราฟ 3 มิติ)
# ==========================================
print("📈 Generating 3D Trajectory Plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. พล็อตเส้นทางของแก้ว (เส้นประสีเทา)
ax.plot(cup_x, cup_y, cup_z, 'k--', alpha=0.5, label='Cup Path')
# จุดที่แก้วโดนจับพอดี (สีแดง)
ax.scatter(cup_x[step_B], cup_y[step_B], cup_z[step_B], color='red', s=50, label='Cup at Grab Step')

# 2. พล็อตเส้นทางของปลายมือ EE (เส้นทึบสีน้ำเงิน)
ax.plot(ef_x, ef_y, ef_z, 'b-', linewidth=2, label='EE Trajectory')

# 3. มาร์คจุดสำคัญต่างๆ ให้ดูง่ายๆ
ax.scatter(*p_home, color='green', s=100, marker='o', label='1. Home')
ax.scatter(*p_via, color='orange', s=80, marker='^', label='2. Via Point (ข้ามสายพาน)')
ax.scatter(*p_wait, color='purple', s=80, marker='s', label='3. Wait Point (ดักรอ)')
ax.scatter(*p_grab_start, color='red', s=150, marker='X', label='4. Grab (ตะปบ)')
ax.scatter(*p_lift_end, color='cyan', s=100, marker='*', label='5. Lift (ยกชูฟ้า)')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Robotic Arm Side-Grasp Trajectory Planning')
ax.legend()

plt.tight_layout()
plt.savefig('trajectory_3d_plot.png', dpi=300)
print("✅ Plot saved successfully to -> trajectory_3d_plot.png")