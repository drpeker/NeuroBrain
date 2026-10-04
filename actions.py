import hardware

ALLOWED_ACTIONS = {
    "LED_ON",
    "LED_OFF",
}

def execute(action):
    action = action.strip().upper()

    if action not in ALLOWED_ACTIONS:
        return {
            "success": False,
            "action": action,
            "message": "Action not allowed"
        }

    if action == "LED_ON":
        message = hardware.led_on()

    elif action == "LED_OFF":
        message = hardware.led_off()

    return {
        "success": True,
        "action": action,
        "message": message,
        "state": hardware.get_state()
    }
