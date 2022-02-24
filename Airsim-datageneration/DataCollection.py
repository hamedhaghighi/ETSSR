import airsim
import cv2
import numpy as np
import os
import time
import tempfile
import matplotlib.pyplot as plt
import itertools

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
# client.enableApiControl(True)
# car_controls = airsim.CarControls()

# print("API Control enabled: %s" % client.isApiControlEnabled())

def NormalizeandRemoveInf(d):
    d_t = np.copy(d)
    d[d == d.max()] = -1.0
    second_max = d.max()
    d = (d_t - d_t.min()) / (second_max - d_t.min())
    d = np.clip(d, 0.0, 1.0)
    return d

environment = 'AirsimNH'
root_dir = os.path.join('/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR' , environment)
if os.path.isdir(root_dir):
    start_idx = len(os.listdir(root_dir))
else:
    start_idx = 0

for idx in range(start_idx, start_idx + 10):
    # get state of the car
    # car_state = client.getCarState()
    # print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # # go forward
    # car_controls.throttle = 0.5
    # car_controls.steering = 0
    # client.setCarControls(car_controls)
    # print("Go Forward")
    # time.sleep(3)   # let car drive a bit

    # # apply brakes
    # car_controls.brake = 1
    # client.setCarControls(car_controls)
    # print("Apply brakes")
    # time.sleep(3)   # let car drive a bit
    # car_controls.brake = 0 #remove brake
    time.sleep(5)

    requests = []

    for i in range(2):
        cam_i_req = [airsim.ImageRequest(str(i), airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(str(i), airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest(str(i), airsim.ImageType.SurfaceNormals, False, False),
        airsim.ImageRequest(str(i), airsim.ImageType.Segmentation, False, False)
        ]
        requests.extend(cam_i_req)
    responses = client.simGetImages(requests)
    LandR_list = []
    for i in range(2):
        response_list = []
        for j in range(4):
            response = responses[i*4 + j]
            if response.pixels_as_float:
                response = np.expand_dims(airsim.get_pfm_array(response), axis=-1)
                img_d = 0.54 * (response.shape[1]/2) * (1/response)
                response_list.append(img_d)
            else:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) #get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3)
                response_list.append(img_rgb[:,:,::-1].astype('float32'))
        LandR_list.append(np.concatenate(response_list, axis=-1))
    img_path = os.path.join(root_dir, 'img_'+ str(idx))
    os.makedirs(img_path, exist_ok=True)
    filename = os.path.join(img_path, 'im0.npy')
    np.save(filename, LandR_list[0])
    filename = os.path.join(img_path, 'im1.npy')
    np.save(filename, LandR_list[1])
    print('image', idx, 'from environemnt ', environment)

#restore to original state
client.reset()

# client.enableApiControl(False)