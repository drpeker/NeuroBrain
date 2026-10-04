from collections import deque, Counter
import time

class VisionState:
    def __init__(self, history_size=5, min_hits=3):
        self.history = deque(maxlen=history_size)
        self.min_hits = min_hits
        self.state = {
            "timestamp": 0,
            "objects": []
        }

    def update(self, detections, width, height):
        frame_objects = []

        for d in detections:
            x1, y1, x2, y2 = d["box"]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if cx < width / 3:
                horizontal = "left"
            elif cx > width * 2 / 3:
                horizontal = "right"
            else:
                horizontal = "center"

            if cy < height / 3:
                vertical = "upper"
            elif cy > height * 2 / 3:
                vertical = "lower"
            else:
                vertical = "middle"

            area = ((x2-x1) * (y2-y1)) / (width * height)

            if area > 0.30:
                distance_hint = "near"
            elif area > 0.08:
                distance_hint = "medium"
            else:
                distance_hint = "far"

            frame_objects.append({
                "name": d["name"],
                "confidence": round(d["confidence"], 2),
                "horizontal": horizontal,
                "vertical": vertical,
                "distance_hint": distance_hint,
                "box": [
                    int(x1), int(y1),
                    int(x2), int(y2)
                ]
            })

        self.history.append(frame_objects)

        counts = Counter()

        for frame in self.history:
            for obj in set(o["name"] for o in frame):
                counts[obj] += 1

        stable = []

        current_by_name = {}

        for obj in frame_objects:
            name = obj["name"]

            if (
                name not in current_by_name
                or obj["confidence"] >
                   current_by_name[name]["confidence"]
            ):
                current_by_name[name] = obj

        for name, obj in current_by_name.items():
            if counts[name] >= self.min_hits:
                stable.append(obj)

        stable.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        self.state = {
            "timestamp": time.time(),
            "objects": stable
        }

        return self.state

    def get(self):
        return self.state

    def summary(self):
        objects = self.state["objects"]

        if not objects:
            return "I currently see no stable objects."

        parts = []

        for o in objects:
            parts.append(
                f'{o["name"]} '
                f'at {o["horizontal"]}-{o["vertical"]} '
                f'({o["distance_hint"]})'
            )

        return "I currently see: " + ", ".join(parts) + "."
