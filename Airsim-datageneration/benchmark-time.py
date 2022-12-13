import time

import airsim

stereo_camera = False
client = airsim.CarClient()
client.confirmConnection()
for i in range(100):
    for j in range(2):
        start = time.time()
        if stereo_camera:
            responses = client.simGetImages(
                [
                    airsim.ImageRequest(
                        '0',
                        airsim.ImageType.Scene,
                        False,
                        False),
                    airsim.ImageRequest(
                        '1',
                        airsim.ImageType.Scene,
                        False,
                        False)])
        else:
            responses = client.simGetImages(
                [airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])
        print(
            'num of cams:',
            int(stereo_camera) + 1,
            'time: ',
            time.time() - start)
        stereo_camera = not stereo_camera
