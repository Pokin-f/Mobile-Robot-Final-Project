from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("📡 Connecting to CoppeliaSim to extract frames & parameters...")
client = RemoteAPIClient()
sim = client.require('sim')

# ==========================================
# 1. GET HANDLES
# ==========================================
joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
ef_handle = sim.getObjectHandle('/yaskawa/gripperEF')
base_handle = sim.getObjectParent(joint_handles[0])

# ==========================================
# 2. FUNCTION ถอดสมการ & พรินต์พารามิเตอร์
# ==========================================
def get_T(handle, ref=-1):
    m = sim.getObjectMatrix(handle, ref)
    return np.array([
        [m[0], m[1], m[2], m[3]],
        [m[4], m[5], m[6], m[7]],
        [m[8], m[9], m[10], m[11]],
        [0,    0,    0,    1]
    ])

def print_kinematic_params(name, handle, ref=-1):
    pos = sim.getObjectPosition(handle, ref)
    ori = sim.getObjectOrientation(handle, ref)
    
    # แปลงเป็น มิลลิเมตร และ องศา เพื่อให้มนุษย์อ่านง่าย
    px, py, pz = np.array(pos) * 1000
    rx, ry, rz = np.degrees(ori)
    
    print(f"{name:<10} | {px:>8.1f} {py:>8.1f} {pz:>8.1f} | {rx:>10.1f} {ry:>10.1f} {rz:>10.1f}")

# ==========================================
# 3. 📊 พิมพ์ตาราง PARAMETERS ลง CONSOLE
# ==========================================
print("\n" + "="*70)
print("🤖 KINEMATIC PARAMETERS (World Frame Coordinates)")
print("="*70)
print(f"{'Frame':<10} | {'X (mm)':<8} {'Y (mm)':<8} {'Z (mm)':<8} | {'Roll (deg)':<10} {'Pitch (deg)':<10} {'Yaw (deg)':<10}")
print("-" * 70)

print_kinematic_params("Base", base_handle)
for i, h in enumerate(joint_handles):
    print_kinematic_params(f"Joint {i+1}", h)
print_kinematic_params("End-Eff", ef_handle)
print("=" * 70 + "\n")

# ==========================================
# 4. วาดแกน 3D (X=Red, Y=Green, Z=Blue)
# ==========================================
def plot_frame(ax, T, name="", scale=0.1):
    origin = T[:3, 3]               
    x_axis = origin + T[:3, 0] * scale  
    y_axis = origin + T[:3, 1] * scale  
    z_axis = origin + T[:3, 2] * scale  

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', linewidth=2)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', linewidth=2)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', linewidth=2)

    if name:
        ax.text(origin[0], origin[1], origin[2], name, fontsize=10, fontweight='bold')
    return origin

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
origins = []

T_base = get_T(base_handle)
origins.append(plot_frame(ax, T_base, "Base", scale=0.2))

for i, h in enumerate(joint_handles):
    T_j = get_T(h)
    origins.append(plot_frame(ax, T_j, f"J{i+1}"))

T_ef = get_T(ef_handle)
origins.append(plot_frame(ax, T_ef, "EF", scale=0.15))

origins = np.array(origins)
ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], color='k', linestyle='--', linewidth=2, label='Robot Links')

# จัดหน้าตากราฟ
ax.set_xlabel('World X')
ax.set_ylabel('World Y')
ax.set_zlabel('World Z')
ax.set_title('Robot Coordinate Frames & Kinematic Chain')

max_range = np.array([origins[:,0].max()-origins[:,0].min(), origins[:,1].max()-origins[:,1].min(), origins[:,2].max()-origins[:,2].min()]).max() / 2.0
mid_x = (origins[:,0].max()+origins[:,0].min()) * 0.5
mid_y = (origins[:,1].max()+origins[:,1].min()) * 0.5
mid_z = (origins[:,2].max()+origins[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.legend()
plt.savefig('robot_frames.png', dpi=150)
print("✅ Successfully saved 3D frame plot to -> robot_frames.png")