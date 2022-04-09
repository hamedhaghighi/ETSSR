from cv2 import cvtColor
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
car_controls = airsim.CarControls()

# print("API Control enabled: %s" % client.isApiControlEnabled())

def NormalizeandRemoveInf(d):
    d_t = np.copy(d)
    d[d == d.max()] = -1.0
    second_max = d.max()
    d = (d_t - d_t.min()) / (second_max - d_t.min())
    d = np.clip(d, 0.0, 1.0)
    return d

def downsample(img):
    img_d = cv2.resize(img.astype('uint8'), (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    img_d = img_d.astype('float32')
    img_d[:, :, 3] = cv2.resize(img[:, :, 3], (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])))
    return img_d

def save_sensor_data(filename, data):
    if len(data.shape) == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, data)
    elif len(data.shape) == 2:
        np.savez_compressed(filename, a=data)


def downsample(img, scale):
    if scale > 1:
        if len(img.shape) == 3:
            img = cv2.resize(img, (int(
                (1/scale) * img.shape[1]), int(1/scale * img.shape[0])), interpolation=cv2.INTER_CUBIC)
        elif len(img.shape) == 2:
            img = cv2.resize(
                img, (int((1/scale) * img.shape[1]), int((1/scale) * img.shape[0])))
            img = img / scale
    return img


environment = 'AirSimNH'
root_dir = os.path.join('/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR/data/AirSim2' , environment)
if os.path.isdir(root_dir):
    print('root dir exists, exitting ...')
    os._exit(1)
    # start_idx = len(os.listdir(root_dir))

for idx in range(100):

    car_state = client.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # go forward
    car_controls.throttle = 2
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
    responses = [client.simGetImages([r])[0] for r in requests]
    # responses = [client.simGetImages(requests)]
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
    for scale in [1 ,2 ,4]:
        for n_cam in [0, 1]:
            img_path = os.path.join(root_dir, 'img_'+ str(idx))
            os.makedirs(img_path, exist_ok=True)
            filename = os.path.join(img_path, 'imgx{}_{}.png'.format(scale, n_cam))
            save_sensor_data(filename, downsample(LandR_list[n_cam][..., :3], scale))
            filename = os.path.join(img_path, 'imgx{}_disp_{}'.format(scale, n_cam))
            save_sensor_data(filename, downsample(LandR_list[n_cam][..., 3], scale))
            filename = os.path.join(img_path, 'imgx{}_sn_{}.png'.format(scale, n_cam))
            save_sensor_data(filename, downsample(LandR_list[n_cam][..., 4:7], scale))
            filename = os.path.join(img_path, 'imgx{}_seg_{}.png'.format(scale, n_cam))
            save_sensor_data(filename, downsample(LandR_list[n_cam][..., 7:], scale))
    print('image', idx, 'from environemnt ', environment)

#restore to original state
client.reset()

# client.enableApiControl(False)
