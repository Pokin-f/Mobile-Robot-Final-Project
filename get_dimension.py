from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

client = RemoteAPIClient()
sim    = client.getObject('sim')

# handles ของทุก link
handles = {
    'base'   : sim.getObjectHandle('/yaskawa/base_link_base'),
    'joint1' : sim.getObjectHandle('/yaskawa/joint1'),
    'link1'  : sim.getObjectHandle('/yaskawa/link_1_s_respondable'),
    'joint2' : sim.getObjectHandle('/yaskawa/joint2'),
    'link2'  : sim.getObjectHandle('/yaskawa/link_2_l_respondable'),
    'joint3' : sim.getObjectHandle('/yaskawa/joint3'),
    'link3'  : sim.getObjectHandle('/yaskawa/link_3_u_respondable'),
    'joint4' : sim.getObjectHandle('/yaskawa/joint4'),
    'link4'  : sim.getObjectHandle('/yaskawa/link_4_r_respondable'),
    'joint5' : sim.getObjectHandle('/yaskawa/joint5'),
    'link5'  : sim.getObjectHandle('/yaskawa/link_5_b_respondable'),
    'joint6' : sim.getObjectHandle('/yaskawa/joint6'),
    'link6'  : sim.getObjectHandle('/yaskawa/link_6_t_respondable'),
    'ef'     : sim.getObjectHandle('/yaskawa/gripperEF'),
}

sim.setStepping(True)
sim.startSimulation()
sim.step()

print("=== Position of each frame relative to WORLD ===")
for name, h in handles.items():
    pos = sim.getObjectPosition(h, -1)
    print(f"{name:<10}: x={pos[0]:8.5f}, y={pos[1]:8.5f}, z={pos[2]:8.5f}")

print("\n=== Link lengths (distance between joints) ===")
joint_handles = [handles[f'joint{i}'] for i in range(1, 7)]
joint_handles.insert(0, handles['base'])
joint_handles.append(handles['ef'])

names = ['base', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'ef']

for i in range(len(joint_handles)-1):
    p1 = np.array(sim.getObjectPosition(joint_handles[i],   -1))
    p2 = np.array(sim.getObjectPosition(joint_handles[i+1], -1))
    diff = p2 - p1
    dist = np.linalg.norm(diff)
    print(f"{names[i]:<6} → {names[i+1]:<6}: "
          f"dx={diff[0]:7.4f} dy={diff[1]:7.4f} dz={diff[2]:7.4f} "
          f"| dist={dist*1000:.2f} mm")

print("\n=== Axis direction of each joint (z-axis = rotation axis) ===")
for i in range(1, 7):
    h   = handles[f'joint{i}']
    m   = sim.getObjectMatrix(h, -1)
    z   = [m[2], m[6], m[10]]
    x   = [m[0], m[4], m[8]]
    print(f"joint{i}: z={np.round(z,3)}, x={np.round(x,3)}")

sim.stopSimulation()