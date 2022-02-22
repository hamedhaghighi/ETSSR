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
client.enableApiControl(True)

print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
print ("Saving images to %s" % tmp_dir)

def NormalizeandRemoveInf(d):
    d_t = np.copy(d)
    d[d == d.max()] = -1.0
    second_max = d.max()
    d = (d_t - d_t.min()) / (second_max - d_t.min())
    d = np.clip(d, 0.0, 1.0)
    return d

try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise
root_dir = '/media/oem/Local Disk/Phd-datasets/iPASSR/datasets_Airsim'

for idx in range(10):
    # get state of the car
    car_state = client.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # go forward
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)
    print("Go Forward")
    time.sleep(3)   # let car drive a bit

    # apply brakes
    car_controls.brake = 1
    client.setCarControls(car_controls)
    print("Apply brakes")
    time.sleep(3)   # let car drive a bit
    car_controls.brake = 0 #remove brake

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
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3)
                response_list.append(img_rgb[:,:,::-1].astype('float32'))
        LandR_list.append(np.concatenate(response_list, axis=-1))
    img_path = os.path.join(root_dir, 'img_'+ str(idx))
    os.makedirs(img_path, exist_ok=True)
    filename = os.path.join(img_path, 'im0.npy')
    np.save(filename, LandR_list[0])
    filename = os.path.join(img_path, 'im1.npy')
    np.save(filename, LandR_list[1])

#restore to original state
client.reset()

client.enableApiControl(False)