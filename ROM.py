#find rangee of motion of each joint prevent from going to singularity or gimbal lock

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

client = RemoteAPIClient()
sim    = client.getObject('sim')

joint_handles = []
for i in range(1, 7):
    h = sim.getObjectHandle(f'/yaskawa/joint{i}')
    joint_handles.append(h)

sim.setStepping(True)
sim.startSimulation()
sim.step()

print("=== Joint Range of Motion ===")
for i, h in enumerate(joint_handles):
    # ดึง joint limits
    result = sim.getJointInterval(h)
    cyclic = result[0]
    interval = result[1]
    
    if cyclic:
        print(f"joint{i+1}: CYCLIC (ไม่มี limit)")
    else:
        lo  = interval[0]
        hi  = interval[0] + interval[1]
        print(f"joint{i+1}: "
              f"min={np.degrees(lo):8.2f}° ({lo:.4f} rad)  "
              f"max={np.degrees(hi):8.2f}° ({hi:.4f} rad)  "
              f"range={np.degrees(interval[1]):.2f}°")

sim.stopSimulation()