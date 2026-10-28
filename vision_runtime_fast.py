from vision_state import VisionState
from vision_service import VisionService
import cv2
cv2.setNumThreads(1)
import ncnn
import numpy as np
import time

PARAM = "/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.param"
BIN   = "/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.bin"

TARGET = 320

# Fast version: YOLO runs on one core, at most twice a second, and pauses while
# NeuroBrain transcribes, thinks and speaks. Running non-stop on 1+ cores made
# Qwen (which wants all 4 cores) 4-5x slower.
VISION_PERIOD = 0.5
CONF = 0.35
NMS_THRESHOLD = 0.45

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

_GRIDS = {}


def _grid(input_w, input_h):
    """Cell centres and strides for every output row, cached per input size."""
    key = (input_w, input_h)
    if key not in _GRIDS:
        cx, cy, st = [], [], []
        for stride in (8, 16, 32):
            gw, gh = input_w // stride, input_h // stride
            gy, gx = np.mgrid[0:gh, 0:gw]
            cx.append(((gx.ravel() + 0.5) * stride))
            cy.append(((gy.ravel() + 0.5) * stride))
            st.append(np.full(gw * gh, stride))
        _GRIDS[key] = tuple(np.concatenate(a).astype(np.float32) for a in (cx, cy, st))
    return _GRIDS[key]


def decode_detections(pred, input_w, input_h,
                      scale, left, top,
                      orig_w, orig_h):
    """Decode YOLOv8 output rows into boxes, all rows at once with numpy.

    The old per-row Python loop took ~50 ms per frame; this takes ~2 ms.
    """

    cx, cy, stride = _grid(input_w, input_h)

    logits = pred[:, 64:144]
    class_ids = np.argmax(logits, axis=1)
    confidences = 1.0 / (1.0 + np.exp(-logits[np.arange(len(pred)), class_ids]))

    keep = confidences >= CONF
    if not np.any(keep):
        return []

    # YOLOv8 DFL: 4 sides x 16 bins, expected value of a softmax
    reg = pred[keep, :64].reshape(-1, 4, 16)
    reg = np.exp(reg - reg.max(axis=2, keepdims=True))
    reg /= reg.sum(axis=2, keepdims=True)
    dist = (reg * np.arange(16, dtype=np.float32)).sum(axis=2) * stride[keep, None]

    x1 = (cx[keep] - dist[:, 0] - left) / scale
    y1 = (cy[keep] - dist[:, 1] - top) / scale
    x2 = (cx[keep] + dist[:, 2] - left) / scale
    y2 = (cy[keep] + dist[:, 3] - top) / scale

    x1 = np.clip(x1, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y2 = np.clip(y2, 0, orig_h - 1)

    detections = []

    for i, class_id, confidence in zip(range(len(x1)), class_ids[keep], confidences[keep]):

        if x2[i] <= x1[i] or y2[i] <= y1[i]:
            continue

        detections.append({
            "class_id": int(class_id),
            "name": CLASSES[class_id],
            "confidence": float(confidence),
            "box": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])]
        })

    return detections


def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return intersection / (
        area_a + area_b - intersection + 1e-6
    )


def nms(detections):

    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    result = []

    while detections:

        best = detections.pop(0)
        result.append(best)

        remaining = []

        for d in detections:

            # Different classes do not suppress each other
            if d["class_id"] != best["class_id"]:
                remaining.append(d)
                continue

            if box_iou(d["box"], best["box"]) <= NMS_THRESHOLD:
                remaining.append(d)

        detections = remaining

    return result



import threading


class VisionRuntime:

    def __init__(self):
        self.service = VisionService()
        self.vision = VisionState(history_size=5, min_hits=3)

        self._stop_event = threading.Event()
        self._thread = None
        self._cap = None
        self._net = None
        self._paused = threading.Event()
        self._next_at = 0.0


    def start(self):

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="NeuroBrainVision",
            daemon=True
        )

        self._thread.start()


    def stop(self):

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)


    def pause(self):
        """Stop running YOLO (frames are still grabbed so the camera buffer stays fresh)."""
        self._paused.set()


    def resume(self):
        self._paused.clear()


    def get_state(self):
        return self.service.get_state()


    def get_objects(self):
        return self.service.get_objects()


    def summary(self):
        return self.service.summary()


    def age(self):
        return self.service.age()


    def is_fresh(self, max_age=2.0):
        return self.service.is_fresh(max_age)


    def wait_fresh(self, max_age=1.0, timeout=2.0):
        """Wait until YOLO has published a state no older than max_age."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.service.is_fresh(max_age):
                return True
            time.sleep(0.05)
        return False


    def _run(self):

        net = ncnn.Net()
        net.opt.num_threads = 1
        net.opt.use_vulkan_compute = False

        if net.load_param(PARAM) != 0:
            print("VISION ERROR: unable to load YOLO param")
            return

        if net.load_model(BIN) != 0:
            print("VISION ERROR: unable to load YOLO model")
            return

        self._net = net

        print("Vision: YOLOv8n loaded.")

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("VISION ERROR: unable to open C270")
            return

        self._cap = cap

        print("Vision: C270 opened.")

        try:

            while not self._stop_event.is_set():

                # Paused or not yet time: grab and drop the frame so the buffer stays fresh
                if self._paused.is_set() or time.monotonic() < self._next_at:
                    cap.grab()
                    continue
                self._next_at = time.monotonic() + VISION_PERIOD

                ok, frame = cap.read()

                if not ok:
                    continue

                orig_h, orig_w = frame.shape[:2]

                # Aspect-ratio preserving resize

                if orig_w > orig_h:

                    scale = TARGET / orig_w

                    new_w = TARGET
                    new_h = int(orig_h * scale)

                else:

                    scale = TARGET / orig_h

                    new_h = TARGET
                    new_w = int(orig_w * scale)

                resized = cv2.resize(
                    frame,
                    (new_w, new_h)
                )

                # Pad to multiple of 32

                wpad = ((new_w + 31) // 32) * 32 - new_w
                hpad = ((new_h + 31) // 32) * 32 - new_h

                left = wpad // 2
                right = wpad - left

                top = hpad // 2
                bottom = hpad - top

                padded = cv2.copyMakeBorder(
                    resized,
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=(114,114,114)
                )

                input_h, input_w = padded.shape[:2]

                # NCNN input

                mat = ncnn.Mat.from_pixels(
                    padded,
                    ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                    input_w,
                    input_h
                )

                mat.substract_mean_normalize(
                    [],
                    [1/255.0, 1/255.0, 1/255.0]
                )

                ex = net.create_extractor()
                ex.input("in0", mat)

                ret, out = ex.extract("out0")

                if ret != 0:
                    continue

                pred = np.array(out)

                detections = decode_detections(
                    pred,
                    input_w,
                    input_h,
                    scale,
                    left,
                    top,
                    orig_w,
                    orig_h
                )

                detections = nms(detections)

                state = self.vision.update(
                    detections,
                    orig_w,
                    orig_h
                )

                self.service.publish(state)

        except Exception as e:

            print("VISION ERROR:", e)

        finally:

            cap.release()
            self._cap = None

            print("Vision: C270 released.")
