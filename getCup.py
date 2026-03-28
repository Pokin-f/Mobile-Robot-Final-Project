from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt

print("📡 Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# หา Object แก้ว
cup_handle = sim.getObjectHandle('/conveyorSystem/Cup')

# เตรียมพื้นที่จดบันทึก
t_list, x_list, y_list, z_list = [], [], [], []
roll_list, pitch_list, yaw_list = [], [], []

dt = 0.05
sim.setFloatParameter(sim.floatparam_simulation_time_step, dt)
sim.setStepping(True)
sim.startSimulation()

print("⏳ Tracking Cup... (รอแก้ววนกลับมาจุดเริ่มต้น)")

# 🌟 1. จำพิกัดเริ่มต้นไว้ก่อน
start_pos = sim.getObjectPosition(cup_handle, sim.handle_world)
start_xy = np.array([start_pos[0], start_pos[1]])
left_start_zone = False # สถานะบอกว่าแก้ววิ่งออกไปไกลหรือยัง

# ใช้ while loop แทนการจำกัด steps
step = 0
max_failsafe_steps = 2000 # กันเหนียวไว้เผื่อสายพานพัง จะได้ไม่ลูปค้างตลอดกาล

while step < max_failsafe_steps:
    t = step * dt
    pos = sim.getObjectPosition(cup_handle, sim.handle_world)
    ori = sim.getObjectOrientation(cup_handle, sim.handle_world)
    
    t_list.append(t)
    x_list.append(pos[0])
    y_list.append(pos[1])
    z_list.append(pos[2])
    roll_list.append(ori[0])
    pitch_list.append(ori[1])
    yaw_list.append(ori[2])
    
    # 🌟 2. คำนวณระยะห่างจากจุดเริ่มต้น (วัดแค่แกน X, Y)
    current_xy = np.array([pos[0], pos[1]])
    dist_to_start = np.linalg.norm(current_xy - start_xy)
    
    # ถ้าแก้ววิ่งออกไปไกลกว่า 30 ซม. แล้ว ให้ถือว่าออกจากจุดสตาร์ทแล้ว
    if not left_start_zone and dist_to_start > 0.30:
        left_start_zone = True
        
    # ถ้าออกจากจุดสตาร์ทไปแล้ว และ "วนกลับมา" ใกล้จุดสตาร์ทน้อยกว่า 5 ซม. -> แปลว่าครบรอบ!
    if left_start_zone and dist_to_start < 0.05:
        print(f"🏁 แก้ววนครบรอบพอดีเป๊ะที่ Step {step} (เวลา {t:.2f} วิ)!")
        break # 🌟 สั่งหยุด Loop ทันที!
        
    sim.step()
    step += 1

sim.stopSimulation()
print("✅ Tracking finished & Simulation Stopped!")

# บันทึกข้อมูล
np.save('cup_trajectory.npy', {
    't': t_list, 'x': x_list, 'y': y_list, 'z': z_list,
    'roll': roll_list, 'pitch': pitch_list, 'yaw': yaw_list
})
print("💾 Saved perfectly looped data to -> cup_trajectory.npy")

# พล็อตดูกราฟ
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_list, y_list, z_list, 'r-', linewidth=2, label='Cup Path (Full Loop)')
ax.scatter(x_list[0], y_list[0], z_list[0], c='green', s=100, label='Start/End Point')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Step 1: Clean Cup Trajectory (Auto-Stop)')
ax.legend()
plt.tight_layout()
plt.savefig('cup_path_check.png')
print("📈 Saved plot to cup_path_check.png")