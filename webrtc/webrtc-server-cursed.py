import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from torchvision import transforms
from hashlib import sha1

from PIL import Image

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder


import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

idx = 0

# cursed routines
def load_checkpoints(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector




class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        global generator
        global kp_detector
        global source
        global kp_source
        global kp1
        global idx

        frame = await self.track.recv()
        idx+=1
        

        if self.transform == "cartoon":
            img = frame.to_ndarray(format="rgb24")
            rimg = cv2.resize(img,(int(342),int(256))) # downsample 640x480 to 342x256 while preserving the AR

            simg = rimg[0:0+256, 43:43+256]

            # sharpen?
            #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            #simg = cv2.filter2D(simg, -1, kernel)

            # enhance?
            #lab= cv2.cvtColor(simg, cv2.COLOR_BGR2LAB)
            #l, a, b = cv2.split(lab)
            #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            #cl = clahe.apply(l)
            #limg = cv2.merge((cl,a,b))
            #simg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # prepare unprocessed frame
            new_frame = VideoFrame.from_ndarray(simg, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            # driving frame should be of torch.Size([1, 3, 256, 256]) 
            im = torch.tensor(simg[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
            driving_frame = im

            # use 100th frame for kp_driving_initial
            if idx == 100:
                print ('shapes: img', img.shape, 'rimg', rimg.shape, 'simg:', simg.shape )
                kp1 = kp_detector(driving_frame)
                #imageio.imsave(os.path.join('tmp',  'driving.png'), simg)

            #imageio.imsave(os.path.join('tmp',  'real'+str(idx)+'.png'), simg)

            if idx > 100: # apply xform after the 100th frame 
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       #kp_driving_initial=kp_driving_initial, 
                                       kp_driving_initial=kp1,  # None doesnt work when adapt_movement_scale or use_relative_movement set to True
                                       use_relative_movement=False,
                                       use_relative_jacobian=False, 
                                       adapt_movement_scale=False)
                # out['prediction']: torch.Size([1, 3, 256, 256]) dtype torch.float32
                with torch.no_grad():
                    out = generator(source, kp_source=kp_source, kp_driving=kp_norm) #was kp_norm
                    prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                    newimg = (255 * prediction).astype(np.uint8)

                    # just to test conversion of image to tensor and back, like latency of the code
                    #newimg = np.transpose(im.data.cpu().numpy(), [0, 2, 3, 1])[0].astype(np.uint8)
                    #newimg = cv2.resize(newimg,(int(256),int(256)))

                    new_frame = VideoFrame.from_ndarray(newimg, format="rgb24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                
            return new_frame
        
        else:
            return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    #player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    #if args.write_audio:
    #    recorder = MediaRecorder(args.write_audio)
    #else:
    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            #pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            local_video = VideoTransformTrack(
                track, transform=params["video_transform"]
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":

    global generator
    global kp_detector
    global source
    global kp_source

    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None
    kp1 = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)

    # initialize the  model
    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    source_image = imageio.imread('mlma.jpg')
    source_image = resize(source_image, (256, 256))[..., :3]
    print('Loading ckpt')
    generator, kp_detector = load_checkpoints(config_path='config/vox-adv-256.yaml', checkpoint_path='vox-adv-cpk.pth.tar')
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
    kp_source = kp_detector(source)
    print('Loading ckpt done')

    # pass generator, kp_detector, source, kp_source as globals
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
