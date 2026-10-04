import cv2
import ncnn
import numpy as np
import time

PARAM = "/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.param"
BIN   = "/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.bin"

TARGET = 320
CONF = 0.25

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# -------------------------------------------------
# YOLO
# -------------------------------------------------

net = ncnn.Net()
net.opt.num_threads = 4
net.opt.use_vulkan_compute = False

if net.load_param(PARAM) != 0:
    raise SystemExit("Could not load YOLO param")

if net.load_model(BIN) != 0:
    raise SystemExit("Could not load YOLO model")

print("YOLOv8n loaded.")

# -------------------------------------------------
# C270
# -------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise SystemExit("C270 could not be opened")

print("C270 opened.")
print("Starting live NeuroBrain vision...")
print("Ctrl+C to stop.\n")

try:
    while True:

        ok, frame = cap.read()

        if not ok:
            print("Camera frame failed")
            continue

        h, w = frame.shape[:2]

        # Aspect-ratio preserving resize
        if w > h:
            scale = TARGET / w
            nw = TARGET
            nh = int(h * scale)
        else:
            scale = TARGET / h
            nh = TARGET
            nw = int(w * scale)

        resized = cv2.resize(frame, (nw, nh))

        # pad to multiple of 32
        wpad = ((nw + 31) // 32) * 32 - nw
        hpad = ((nh + 31) // 32) * 32 - nh

        left = wpad // 2
        right = wpad - left
        top = hpad // 2
        bottom = hpad - top

        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114,114,114)
        )

        ih, iw = padded.shape[:2]

        mat = ncnn.Mat.from_pixels(
            padded,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            iw,
            ih
        )

        mat.substract_mean_normalize(
            [],
            [1/255.0, 1/255.0, 1/255.0]
        )

        ex = net.create_extractor()
        ex.input("in0", mat)

        t0 = time.perf_counter()

        ret, out = ex.extract("out0")

        elapsed = (time.perf_counter() - t0) * 1000

        if ret != 0:
            print("Inference error:", ret)
            continue

        pred = np.array(out)

        # For the moment inspect strongest class response.
        cls = sigmoid(pred[:,64:144])

        best_row = np.argmax(cls.max(axis=1))
        label = int(np.argmax(cls[best_row]))
        confidence = float(cls[best_row,label])

        if confidence >= CONF:
            status = f"{CLASSES[label]} {confidence:.2f}"
        else:
            status = "nothing"

        print(
            f"{elapsed:6.1f} ms | "
            f"{pred.shape} | "
            f"{status:20s} | "
            f"best={CLASSES[label]} {confidence:.3f}"
        )

except KeyboardInterrupt:
    print("\nStopping vision.")

finally:
    cap.release()
    print("Camera released.")
