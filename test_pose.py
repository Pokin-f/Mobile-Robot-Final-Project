import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

print("📡 Connecting...")
client = RemoteAPIClient()
sim = client.require('sim')
joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]

# =========================================================
# 🎯 ลูกพี่แก้แค่ตัวเลขบรรทัดนี้บรรทัดเดียว! (ใส่เป็นองศาได้เลย)
# =========================================================
# ลำดับข้อต่อ: [J1(ฐาน), J2(ไหล่), J3(ศอก), J4(แขน), J5(ข้อมือ), J6(ปลายมือ)]
test_angles = [0.0, 0.0, 0.0, 0.0, -90.0, 0.0]
q_test = np.radians(test_angles)

sim.setStepping(True)
sim.startSimulation()

# สั่งหุ่นเข้าท่าที่เราเดาไว้
for i, h in enumerate(joint_handles):
    sim.setJointPosition(h, float(q_test[i]))
    sim.setJointTargetPosition(h, float(q_test[i]))

for _ in range(10): sim.step() # ขยับฟิสิกส์ให้ภาพอัปเดต

print(f"👀 หุ่นเข้าท่า {test_angles} แล้ว! ดูที่หน้าจอว่ามือคว่ำลงพื้นสวยมั้ย?")
time.sleep(8) # หยุดให้ลูกพี่ดูผลงาน 8 วินาที

sim.stopSimulation()
print("🛑 Simulation Stopped. หุ่นเด้งกลับท่าเดิม (Scene ปลอดภัย!)")