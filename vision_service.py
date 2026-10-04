import threading
import time


class VisionService:
    """
    NeuroBrain vision interface.

    YOLO/NCNN vision loop can publish the latest stable VisionState here.
    Other NeuroBrain components can safely read it without knowing
    anything about camera, OpenCV or YOLO.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self._state = {
            "timestamp": 0.0,
            "objects": []
        }

    def publish(self, state):
        """
        Publish a new stable vision state.
        """

        if not isinstance(state, dict):
            raise TypeError("Vision state must be a dictionary")

        objects = state.get("objects", [])

        with self._lock:
            self._state = {
                "timestamp": state.get(
                    "timestamp",
                    time.time()
                ),
                "objects": [
                    dict(obj) for obj in objects
                ]
            }

    def get_state(self):
        """
        Return a safe copy of the current vision state.
        """

        with self._lock:
            return {
                "timestamp": self._state["timestamp"],
                "objects": [
                    dict(obj)
                    for obj in self._state["objects"]
                ]
            }

    def get_objects(self):
        return self.get_state()["objects"]

    def summary(self):
        state = self.get_state()
        objects = state["objects"]

        if not objects:
            return "I currently see no stable objects."

        parts = []

        for obj in objects:
            name = obj.get("name", "object")
            horizontal = obj.get("horizontal", "unknown")
            vertical = obj.get("vertical", "unknown")
            distance = obj.get("distance_hint", "unknown")

            parts.append(
                f"{name} at "
                f"{horizontal}-{vertical} "
                f"({distance})"
            )

        return "I currently see: " + ", ".join(parts) + "."

    def age(self):
        """
        Age of current state in seconds.
        """

        state = self.get_state()

        if state["timestamp"] <= 0:
            return None

        return time.time() - state["timestamp"]

    def is_fresh(self, max_age=2.0):
        age = self.age()

        return (
            age is not None
            and age <= max_age
        )
