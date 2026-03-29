from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import sympy as sp

print("📡 Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# ======================================================================
# 🤖 KINEMATIC PARAMETERS (World Frame Coordinates)
# ======================================================================
# Frame      | X (mm)   Y (mm)   Z (mm)   | Roll (deg) Pitch (deg) Yaw (deg) 
# ----------------------------------------------------------------------
# Base       |      0.0      0.0     99.5 |        0.0      -90.0        0.0
# Joint 1    |     18.7      0.2    330.0 |        0.0       -0.0        0.0
# Joint 2    |     58.7      0.2    330.0 |      -90.0       -0.0       -0.0
# Joint 3    |     58.7      0.2    675.0 |       90.0       -0.0        0.0
# Joint 4    |    398.7      0.2    715.0 |        0.0      -90.0        0.0
# Joint 5    |    398.7      0.2    715.0 |       90.0       -0.0        0.0
# Joint 6    |    398.7      0.2    715.0 |        0.0      -90.0        0.0
# End-Eff    |    640.0      0.2    715.1 |      -88.7       89.1      -92.3
# ======================================================================
# พิกัด Absolute (mm) -> แปลงเป็น เมตร (m)
# J1[18.7, 0.2, 330.0], J2[58.7, 0.2, 330.0], ...
# EE[640.0, 0.2, 715.1]

# สกัดระยะ DH Parameters เป๊ะๆ ระดับโรงงาน:
d1 = 330.0 / 1000.0             # ความสูงฐาน (J2 Z อิงจาก World)
a1 = (58.7 - 18.7) / 1000.0     # ระยะเยื้อง J1-J2 (X diff)
a2 = (675.0 - 330.0) / 1000.0   # ความยาวแขนท่อนบน (J3-J2 Z diff)
a3 = (715.0 - 675.0) / 1000.0   # ระยะเยื้องข้อศอก (J4-J3 Z diff)
d4 = (398.7 - 58.7) / 1000.0    # ความยาวท่อนแขนล่าง (J4-J3 X diff)
d6 = (640.0 - 398.7) / 1000.0   # ระยะปลายนิ้ว (EE-J6 X diff)

# ==========================================
# 1. ประกาศตัวแปร Sympy สำหรับทำ Jacobian ตอนท้าย
# ==========================================
q1_s, q2_s, q3_s, q4_s, q5_s, q6_s = sp.symbols('q1 q2 q3 q4 q5 q6')
q_sym = sp.Matrix([q1_s, q2_s, q3_s, q4_s, q5_s, q6_s])

# ==========================================
# 2. ปริ้นท์ DH Table (Numeric/Standard)
# ==========================================
print("\n" + "="*80)
print("📊 1. Standard DH Table (Yaskawa GP8) - วิเคราะห์จากข้อมูล CAD")
print("="*80)
print("| Link i | alpha_{i-1} (rad) | a_{i-1} (m) | d_i (m) | theta_i (rad)     |")
print("-" * 80)
print(f"|   1    |    -1.5708 (-pi/2) |  {a1:.4f}     | {d1:.4f}  | q1                |")
print(f"|   2    |     0.0000         |  {a2:.4f}     | 0.0000  | q2 - 1.5708 (-pi/2)|") 
print(f"|   3    |    -1.5708 (-pi/2) |  {a3:.4f}     | 0.0000  | q3                |")
print(f"|   4    |     1.5708 (pi/2)  |  0.0000     | {d4:.4f}  | q4                |")
print(f"|   5    |    -1.5708 (-pi/2) |  0.0000     | 0.0000  | q5                |")
print(f"|   6    |     0.0000         |  0.0000     | {d6:.4f}  | q6                |")
print("="*80)

# ==========================================
# 3. เตรียมฟังก์ชันเมทริกซ์ (ทั้ง Numeric และ Symbolic)
# ==========================================
def get_dh_numeric(alpha, a, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def get_dh_symbolic(alpha, a, d, theta):
    T = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])
    return sp.nsimplify(T, tolerance=1e-10)

# ==========================================
# 4. อ่านมุมปัจจุบันจาก Sim และคำนวณเมทริกซ์ย่อย (Numeric)
# ==========================================
j_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
ef_handle = sim.getObject('/yaskawa/gripperEF')
q_act = [sim.getJointPosition(h) for h in j_handles]

# ชดเชยระยะฐาน (เยื้อง X 18.7 mm) เทียบกับพิกัดโลก
T_Base_N = np.array([
    [1, 0, 0, 0.0187],
    [0, 1, 0, 0.0000],
    [0, 0, 1, 0.0000],
    [0, 0, 0, 1.0000]
])

T01_N = get_dh_numeric(-np.pi/2, a1, d1, q_act[0])
T12_N = get_dh_numeric(0,        a2, 0,  q_act[1] - np.pi/2)
T23_N = get_dh_numeric(-np.pi/2, a3, 0,  q_act[2])
T34_N = get_dh_numeric(np.pi/2,  0,  d4, q_act[3])
T45_N = get_dh_numeric(-np.pi/2, 0,  0,  q_act[4])
T56_N = get_dh_numeric(0,        0,  d6, q_act[5])

matrices_N = [("T01", T01_N), ("T12", T12_N), ("T23", T23_N), 
              ("T34", T34_N), ("T45", T45_N), ("T56", T56_N)]

# ==========================================
# 5. ปริ้นท์เมทริกซ์ย่อย (Numeric)
# ==========================================
np.set_printoptions(precision=4, suppress=True)
def clean_n(mat):
    m = mat.copy()
    m[np.abs(m) < 1e-5] = 0.0
    return m

print("\n🤖 2. Transformation Matrices ย่อย (Numeric - ที่มุมปัจจุบัน):")
print("-" * 60)
for name, mat in matrices_N:
    print(f"[{name}] =")
    print(clean_n(mat))
    print("-" * 40)

# ==========================================
# 6. คำนวณ T06 (ทฤษฎี) และ Get T06 (Sim) มาปริ้นท์เทียบ
# ==========================================
T06_DH_N = T_Base_N @ T01_N @ T12_N @ T23_N @ T34_N @ T45_N @ T56_N

m = sim.getObjectMatrix(ef_handle, sim.handle_world)
T06_Sim_N = np.array([
    [m[0], m[1], m[2],  m[3]],
    [m[4], m[5], m[6],  m[7]],
    [m[8], m[9], m[10], m[11]],
    [0.0,  0.0,  0.0,   1.0]
])

print("\n🧮 3. ผลลัพธ์รวม T06 (ทฤษฎี DH) vs T06 (Get จาก CoppeliaSim World):")
print("-" * 75)
print("🤖 T06_DH (คำนวณ):")
print(clean_n(T06_DH_N))
print("\n🎯 T06_Sim (จากโปรแกรม World Frame):")
print(clean_n(T06_Sim_N))

pos_dh = T06_DH_N[:3, 3]
pos_sim = T06_Sim_N[:3, 3]
pos_err = np.linalg.norm(pos_dh - pos_sim) * 1000

print(f"\n🔍 ตรวจสอบ Position (เมตร): DH=[{pos_dh[0]:.4f}, {pos_dh[1]:.4f}, {pos_dh[2]:.4f}] | Sim=[{pos_sim[0]:.4f}, {pos_sim[1]:.4f}, {pos_sim[2]:.4f}]")
print(f"⚠️ ความคลาดเคลื่อนตำแหน่ง (Position Error) = {pos_err:.4f} มิลลิเมตร")
print("="*80)

# ==========================================
# 7. ลุยทำ Jacobian Matrix (Symbolic)
# ==========================================
print("\n🧮 4. เริ่มต้นโหมดคณิตศาสตร์... ดิฟสมการหา Analytical Jacobian (Jv)")
print("⏳ กำลังเบ่งสมการตรีโกณมิติยาวเหยียด (อาจจะใช้เวลาสักแป๊บ)...")

T01_S = get_dh_symbolic(-sp.pi/2, a1, d1, q1_s)
T12_S = get_dh_symbolic(0,        a2, 0,  q2_s - sp.pi/2)
T23_S = get_dh_symbolic(-sp.pi/2, a3, 0,  q3_s)
T34_S = get_dh_symbolic(sp.pi/2,  0,  d4, q4_s)
T45_S = get_dh_symbolic(-sp.pi/2, 0,  0,  q5_s)
T56_S = get_dh_symbolic(0,        0,  d6, q6_s)

# รวมร่าง T06 แบบ Symbolic
T06_S = sp.simplify(T01_S * T12_S * T23_S * T34_S * T45_S * T56_S)

# เวกเตอร์ตำแหน่งปลายแขน
P = sp.Matrix([T06_S[0, 3], T06_S[1, 3], T06_S[2, 3]])

# สั่ง .jacobian() เพื่อดิฟอัตโนมัติรวดเดียวจบ!
Jv = sp.simplify(P.jacobian(q_sym))

print("\n✅ ดิฟเสร็จเรียบร้อย! โฉมหน้า Analytical Jacobian Matrix (Jv) ขนาด 3x6:")
print("-" * 60)

# ปริ้นท์สมการดิฟทีละแถว (X, Y, Z) โชว์ใน Terminal
axis_names = ['X', 'Y', 'Z']
for i in range(3):
    print(f"📍 แถวที่ {i+1} (ดิฟ Position แกน {axis_names[i]} เทียบกับมุม q1-q6):")
    for j in range(6):
        # sp.pprint(Jv[i, j]) # ถ้าอยากดูแบบวาดรูปสมการสวยๆ เอาคอมเมนต์ออก
        print(f"   J_{axis_names[i]}{j+1} (d{axis_names[i]}/dq{j+1}) = {Jv[i, j]}")
    print("-" * 40)

print("="*80)
print("✅ สำเร็จครบทุกกระบวนความ Kinematics!! พร้อมบวก Inverse Kinematics ต่อแล้วครับลูกพี่!! 🦾🔥")
