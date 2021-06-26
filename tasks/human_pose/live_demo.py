import json
import trt_pose.coco

import torch
import torch2trt
from torch2trt import TRTModule

import cv2
import torchvision.transforms as transforms
import PIL.Image

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

WIDTH = 1280
HEIGHT = 720
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print("Reading TensorRT model.......")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

print("TensorRT model loaded")

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
    
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def execute(frame, frame_orig):
    data = preprocess(frame)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    draw_objects(frame_orig, counts, objects, peaks)
    
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=2,
    ):
    return (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), "
    "width=(int)%d, height=(int)%d, "
    "format=(string)NV12, framerate=(fraction)%d/1 ! "
    "nvvidconv flip-method=%d ! "
    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
    % (
    capture_width,
    capture_height,
    framerate,
    flip_method,
    display_width,
    display_height,))
    
def main():
    capture_width = 1280
    capture_height = 720
    display_width = 1280
    display_height = 720
    framerate = 60
    flip_method = 2

    # Use CSI Camera
    # cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

    # Use USB Camera
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("human_pose")
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # img_counter = 0

    while True:
        ret, frame_orig = cam.read()
        frame = cv2.resize(frame_orig,(224,224))

        execute(frame, frame_orig)
        fps = cam.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame_orig, 'FPS = '+str(fps), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 5)
        cv2.imshow("human_pose", frame_orig)
        if not ret:
            break
        key = cv2.waitKey(1)
        
        # To close the window press "Q"
        if key & 0xFF == ord('q') or key ==27:
            break

    cam.release()
    cv2.destroyAllWindows()    
    
if __name__ == "__main__":
    main()
