'''
This code is setting up the ttaffic lights timings. 

The traffic lights timing is independent of the vehicle motion.

The traffic lights is initially set at Green condition and it will change every 5 seconds 

'''


import time
import math
import struct
from quanser.communications import Stream

from qvl.qlabs import QuanserInteractiveLabs
from qvl.traffic_light import QLabsTrafficLight

# creates a server connection with Quanser Interactive Labs and manages the communications
qlabs = QuanserInteractiveLabs()

print("Connecting to QLabs...")
# trying to connect to QLabs and open the instance we have created - program will end if this fails
try:
    qlabs.open("localhost")
except:
    print("Unable to connect to QLabs")

# traffic light
    
x_offset = 0.13
y_offset = 1.67

TrafficLight0 = QLabsTrafficLight(qlabs)
TrafficLight0.spawn_degrees([2.3 + x_offset, y_offset, 0], [0, 0, 0], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
TrafficLight1 = QLabsTrafficLight(qlabs)
TrafficLight1.spawn_degrees([-2.3 + x_offset, -1 + y_offset, 0], [0, 0, 180], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)

i = 0
startTime =  time.time()
while (True):
    print("TIME:", time.time()-startTime)
    i=i+1
    print (i)

    if i % 2 == 0:
        TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
        TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
    # elif i>=7 :
    #     TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
    else:
        TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
        TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)

   
    time.sleep(5) 


qlabs.close()
print("Done!")

waypointCounter > 500 