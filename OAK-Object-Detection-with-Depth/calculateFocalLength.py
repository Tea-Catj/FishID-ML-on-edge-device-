import depthai as dai
import numpy as np
import sys
import math
from pathlib import Path

# Connect Device
import depthai as dai

with dai.Device() as device:
  #1st way
  #get focal length from calibration 
  calibData = device.readCalibration()
  intrinsicsR = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, 640, 400)
  print('Right mono camera focal length in pixels in calibration:', intrinsicsR[0][0])
  
  #2nd way
  #run getFOV.py to get the horizontal FOV of mono right or left 
  HFOV = 73.75389511488575
  width = 640 # your chosen resolution for mono cam
  #calculate focal length (only need one from one mono cam)
  f_x = width * (1 / (2 * math.tan((HFOV / 2) * (math.pi / 180))))
  print('Right mono camera focal length in pixels in 400_P:', f_x)