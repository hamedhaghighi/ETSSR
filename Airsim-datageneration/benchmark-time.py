import airsim
import cv2
import numpy as np
import os
import time
import tempfile
import matplotlib.pyplot as plt
import itertools
import time

client = airsim.CarClient()
client.confirmConnection()
for i in range(100):
    for j in range(1):
        start = time.time()
        responses = client.simGetImages([airsim.ImageRequest(str(j), airsim.ImageType.Scene, False, False), 
        airsim.ImageRequest(str(j + 1), airsim.ImageType.Scene, False, False)])
        print('Camera_',j, ':', time.time() - start)
    
