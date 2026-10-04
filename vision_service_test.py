from vision_state import VisionState
from vision_service import VisionService
import cv2
import ncnn
import numpy as np
import time

PARAM = "/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.param"
BIN   = "/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.bin"

TARGET = 320
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def decode_detections(pred, input_w, input_h,
                      scale, left, top,
                      orig_w, orig_h):

    detections = []
    offset = 0

    for stride in (8, 16, 32):

        grid_w = input_w // stride
        grid_h = input_h // stride
        count = grid_w * grid_h

        rows = pred[offset:offset + count]
        offset += count

        for index, row in enumerate(rows):

            scores = sigmoid(row[64:144])
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < CONF:
                continue

            # YOLOv8 DFL: 4 sides × 16 bins
            reg = row[:64].reshape(4, 16)

            bins = np.arange(16, dtype=np.float32)

            distances = []

            for side in reg:
                probabilities = softmax(side)
                distances.append(
                    float(np.sum(probabilities * bins)) * stride
                )

            l, t, r, b = distances

            gx = index % grid_w
            gy = index // grid_w

            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride

            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b

            # Remove padding
            x1 = (x1 - left) / scale
            y1 = (y1 - top) / scale
            x2 = (x2 - left) / scale
            y2 = (y2 - top) / scale

            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w - 1, x2))
            y2 = max(0, min(orig_h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class_id": class_id,
                "name": CLASSES[class_id],
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
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


# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

net = ncnn.Net()
net.opt.num_threads = 4
net.opt.use_vulkan_compute = False

if net.load_param(PARAM) != 0:
    raise SystemExit("ERROR loading YOLO param")

if net.load_model(BIN) != 0:
    raise SystemExit("ERROR loading YOLO model")

print("YOLOv8n loaded.")


# --------------------------------------------------
# Open C270
# --------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise SystemExit("ERROR opening C270")

vision = VisionState(history_size=5, min_hits=3)
service = VisionService()
print("C270 opened.")
print("Live multi-object detection started.")
print("Ctrl+C to stop.\n")


try:

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        orig_h, orig_w = frame.shape[:2]

        # ------------------------------------------
        # Aspect-ratio preserving resize
        # ------------------------------------------

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

        # ------------------------------------------
        # Pad to multiple of 32
        # ------------------------------------------

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

        # ------------------------------------------
        # NCNN input
        # ------------------------------------------

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

        start = time.perf_counter()

        ret, out = ex.extract("out0")

        inference_ms = (
            time.perf_counter() - start
        ) * 1000

        if ret != 0:
            continue

        pred = np.array(out)

        # ------------------------------------------
        # Decode + NMS
        # ------------------------------------------

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

        # ------------------------------------------
        # Stable NeuroBrain vision state
        # ------------------------------------------

        state = vision.update(
            detections,
            orig_w,
            orig_h
        )

        service.publish(state)
        print("SERVICE:", service.summary())

        # ------------------------------------------
        # Print useful state
        # ------------------------------------------

        if detections:

            text = []

            for d in detections:

                x1,y1,x2,y2 = d["box"]

                cx = (x1+x2)/2
                cy = (y1+y2)/2

                # Simple spatial description
                if cx < orig_w/3:
                    horizontal = "left"
                elif cx > orig_w*2/3:
                    horizontal = "right"
                else:
                    horizontal = "center"

                if cy < orig_h/3:
                    vertical = "upper"
                elif cy > orig_h*2/3:
                    vertical = "lower"
                else:
                    vertical = "middle"

                text.append(
                    f'{d["name"]} '
                    f'{d["confidence"]:.2f} '
                    f'[{horizontal}-{vertical}]'
                )

            print(
                f"{inference_ms:5.1f} ms | "
                + " | ".join(text)
            )

        else:

            print(
                f"{inference_ms:5.1f} ms | nothing"
            )


except KeyboardInterrupt:

    print("\nStopping vision.")


finally:

    cap.release()

    print("Camera released.")
