import sim  # CoppeliaSim remote API library

def connect():
    sim.simxFinish(-1)  # close any existing connections
    clientID = sim.simxStart('127.0.0.1', 23000, True, True, 5000, 5)
    if clientID == -1:
        print("Failed to connect")
        return None
    print("Connected, clientID:", clientID)
    return clientID

clientID = connect()