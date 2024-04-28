
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

"""
Main code for Vehicle Control, Traffic Sign and Traffic Light Detection
"""


#--------Imports------------
import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg
import pandas as pd

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.products.qcar import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from pal.products.qcar import QCarRealSense



#---------------Added imports for Tensorflow models-------

from PIL import Image
from model import TLClassifier, download_model, load_graph, select_boxes, crop_roi_image,TSClassifier
import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt


#================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. 
tf = 100
startDelay = 1
controllerUpdateRate = 300

global t0, count, signalFlag, v_ref, waypointCounter
waypointCounter = 0
count = 0
signalFlag= 'Go'


# ===== Speed Controller Parameters
# - v_ref: desired velocity in m/s
# - K_p: proportional gain for speed controller
# - K_i: integral gain for speed controller
# - K_d: derivative gain for speed controller


v_ref = 1.25
K_p = 1
K_i = 1
K_d = 0.01

# ===== Steering Controller Parameters
# - enableSteeringControl: whether or not to enable steering control
# - K_stanley: K gain for stanley controller
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableSteeringControl = True
K_stanley = 2
nodeSequence = [10, 2, 4, 14, 20, 22, 10, 2]


#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


#----------------- An utility function---------------------
def compare_elements(x, my_list, ele):

    if len(my_list) < ele:
        return False 
    
    last_elements = my_list[-ele:]

    return np.array_equal(last_elements, np.full(ele, x))

#-----------------Function to calculate slope of the trajectory in x-y plane----------------- 
def calculate_slope(waypoints):
    slopes = []
    print("waypoints hape", waypoints.shape)
    num_waypoints = waypoints.shape[1]
    print("No of waypoints:", num_waypoints)
    
    # Iterate over the waypoints
    for i in range(num_waypoints - 1):
        # Get the current and next waypoint
        current_point = waypoints[:,i]
        next_point = waypoints[:,i + 1]
        
        # Calculate the change in x and y
        delta_x = next_point[0] - current_point[0]
        delta_y = next_point[1] - current_point[1]
        
        # Check for vertical line (avoid division by zero)
        if delta_x == 0:
            slope = float('inf')  # Vertical line, slope is infinity
        else:
            slope = delta_y / delta_x
        
        slopes.append(slope)
    
    return slopes

#Intializing the Tensorflow Models for Traffic Sign & Traffic Light Detection
tlc=TLClassifier() #For Traffic light detection

tsc = TSClassifier()#For Traffic Sign detection

# Initializing the Tensorflow Model for Traffic Light Classification
base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
) 
base_model.trainable = False
inputs = keras.Input(shape=(150, 150, 3))
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
base_model.trainable = True

model_folder = os.path.dirname(__file__)

model_path = os.path.join(model_folder, 'newModel.weights.h5')

# LLoading the model
model.load_weights(model_path)

#region : Initial setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

#Defining some variables
global KILL_THREAD, done, nextSlope, previousSlopes
nextSlope = 0.01
previousSlopes = 0
KILL_THREAD = False
done = False 
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion

#-------Class for Speed Controller-----------
class SpeedController:
    def __init__(self, kp=0, ki=0, kd=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ei = 0
        self.prev_error = v_ref

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        if dt ==0.0:
            dt = 0.001
        de_dt = (e - self.prev_error) / dt
        output  = self.kp * e + self.ki * self.ei + self.kd * de_dt
        self.prev_error = e

        return np.clip(
             output,
            -self.maxThrottle,
            self.maxThrottle
        )
    
#-------Class for Steering Controller-----------
class SteeringController:

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi/6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.slopes = calculate_slope(waypointSequence)

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0
        self.psi_prev = 0

    # ==============  SECTION B -  Steering Control  ====================
    def update(self, p, th, speed, dt):
        global done, nextSlope, previousSlopes, waypointCounter

        nextSlope = self.slopes[self.wpi+1]
        previousSlopes = self.slopes[:self.wpi]
        if self.wpi>1630:
            done = True

        waypointCounter = self.wpi

        wp_1 = self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = self.wp[:, np.mod(self.wpi+1, self.N-1)]
        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p-wp_1, v_uv)

        if s >= v_mag:
            if  self.cyclic or self.wpi < self.N-2:
                self.wpi += 1

        ep = wp_1 + v_uv*s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent-th) 

        self.p_ref = ep
        self.th_ref = tangent

        if dt == 0.0:
            headingError = psi 
        else:
            headingError = psi +  (0.1 * ((psi- self.psi_prev)/dt))
        self.psi_prev = psi
        return np.clip(
            wrap_to_pi(headingError + np.arctan2(self.k*ect,speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle)
    


#-------Seperate Thread for Traffic Sign detection----------- 
def trafficSign():
    global signalFlag, myCam
    global KILL_THREAD
    while (not KILL_THREAD):
        try:
            
            myCam.read_RGB()
            cv2.imshow('My RGB', myCam.imageBufferRGB)
            cv2.waitKey(100)
            img = Image.fromarray(myCam.imageBufferRGB)
            blue, green, red = img.split()
            rgb_image = Image.merge("RGB", (red, green, blue))
            image =  np.asarray(rgb_image.resize((1280, 960)))
            boxes=tsc.detect_multi_object(image,score_threshold=0.1)
            cropped_image=crop_roi_image(image,boxes[0])
            area_box = cropped_image.shape[0] * cropped_image.shape[1]

            if  4000< area_box < 9000 :
                signalFlag = 'Stop' 
                time.sleep(3)
                signalFlag = 'Go'


        except IndexError:
            pass


#-------Seperate Thread for Traffic Lights detection & Classification----------- 
def trafficLight():
    global signalFlag, myCam

    global KILL_THREAD
    while (not KILL_THREAD):
        try:

            myCam.read_RGB()
            cv2.imshow('My RGB1', myCam.imageBufferRGB)
            cv2.waitKey(100)
            img1 = Image.fromarray(myCam.imageBufferRGB)
            blue1, green1, red1 = img1.split()
            rgb_image1 = Image.merge("RGB", (red1, green1, blue1))
            image1 =  np.asarray(rgb_image1.resize((1280, 960)))
            boxes1=tlc.detect_multi_object(image1,score_threshold=0.1)
            cropped_image1=crop_roi_image(image1,boxes1[0])
            area_box1 = cropped_image1.shape[0] * cropped_image1.shape[1]
            rgb_cropped_image = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2RGB)
            det_img = Image.fromarray(rgb_cropped_image)

            resized_image = det_img.resize((150,150))

            img_array = keras.utils.img_to_array(resized_image)
            img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
            
            predictions = model.predict(img_array)
            score = float(keras.ops.sigmoid(predictions[0][0]))
            oriSignal = 'Stop' if score > 0.5 else 'Go'


            if 2000< area_box1 < 4000 :
                signalFlag = 'Stop' if score > 0.5 else 'Go'
            elif signalFlag == 'Stop' and oriSignal == 'Stop':
                signalFlag == 'Stop'
            else:
                signalFlag = 'Go'

        except IndexError:
            pass
    

#-------Seperate Thread for Vehicle Control----------- 
def controlLoop():
    #region controlLoop setup
    global KILL_THREAD, done, v_ref, waypointCounter, t0

    u = 0
    delta = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region Controller initialization
    speedController = SpeedController(
        kp=K_p,
        ki=K_i,
        kd= K_d
    )
    if enableSteeringControl:
        steeringController = SteeringController(
            waypoints=waypointSequence,
            k=K_stanley
        )
    #endregion

    #region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=initialPose)
    else:
        gps = memoryview(b'')
    #endregion

    with qcar, gps:
        t0 = time.time()
        t=0
        while (t < tf+startDelay) and (not KILL_THREAD):

            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion           
            #region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    # print("______GPS WORKING_____")
                    y_gps = np.array([
                        gps.position[0],
                        gps.position[1],
                        gps.orientation[2]
                    ])
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0,0]
                y = ekf.x_hat[1,0]
                th = ekf.x_hat[2,0]
                p = ( np.array([x, y])
                    + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach
            
            
            #region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0

    
            elif signalFlag == 'Stop':
                v_ref = 0
                u = 0
                delta = 0


            elif done == True:
                v_ref = 0
                u =0 
                delta = 0
                KILL_THREAD = True
    
            else:

                if waypointCounter > 5 and nextSlope != previousSlopes[-1]:
                    v_ref = 0.5
                elif nextSlope == float('inf') and not compare_elements(nextSlope, previousSlopes, 30):
                    v_ref = 0.5
                elif  waypointCounter > 500 and  nextSlope == -float('0.0') and not compare_elements(nextSlope, previousSlopes, 30):
                    v_ref = 0.5
                else:
                        
                    v_ref = 1.25

                u = speedController.update(v, v_ref, dt)
                #endregion

                #region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v, dt)
                    # print("Delta", delta)
                else:
                    delta = 0
            qcar.write(u, delta)
            #endregion

            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref-v])
                speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]

                    x_ref = steeringController.p_ref[0]
                    y_ref = steeringController.p_ref[1]
                    th_ref = steeringController.th_ref

                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]

                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])


                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180-th*180/np.pi)

                count = 0

            #endregion
            continue

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Setup and run experiment
if __name__ == '__main__':

    #region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    # Scope for monitoring speed controller
    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='Vehicle Speed Control',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='Vehicle Speed [m/s]', size=14,
        yLim=(0, 1)
    )
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')

    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='Speed Error [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()

    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='Time [s]',
        yLabel='Throttle Command [%]',
        yLim=(-0.3, 0.3)
    )
    speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    if enableSteeringControl:
        steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='Vehicle Steering Control',
            fps=fps
        )

        steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='x Position [m]',
            yLim=(-2.5, 2.5)
        )
        steeringScope.axes[0].attachSignal(name='x_meas')
        steeringScope.axes[0].attachSignal(name='x_ref')

        steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='y Position [m]',
            yLim=(-1, 5)
        )
        steeringScope.axes[1].attachSignal(name='y_meas')
        steeringScope.axes[1].attachSignal(name='y_ref')

        steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Heading Angle [rad]',
            yLim=(-3.5, 3.5)
        )
        steeringScope.axes[2].attachSignal(name='th_meas')
        steeringScope.axes[2].attachSignal(name='th_ref')

        steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='Steering Angle [rad]',
            yLim=(-0.6, 0.6)
        )
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = 'Time [s]'

        steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )

        im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )

        steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
        steeringScope.axes[4].images[0].setImage(image=im)

        referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            # pen={'color': (0 , 0, 255), 'width': 2},

            name='Reference'
        )
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])

        steeringScope.axes[4].attachSignal(name='Estimated', width=2)

        arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    #endregion

    myCam = QCarRealSense(mode='RGB, Depth')
    #region : Setup control thread, then run experiment
    controlThread = Thread(target=controlLoop)
    trafficLightThread = Thread(target=trafficLight)
    trafficSignThread = Thread(target=trafficSign)

    
    
    trafficSignThread.start()
    time.sleep(5)
    trafficLightThread.start()
    time.sleep(5)
    controlThread.start()

    try:

        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        controlThread.join()
        trafficLightThread.join()
        trafficSignThread.join()

    input('Experiment complete. Press any key to exit...')
#endregion
