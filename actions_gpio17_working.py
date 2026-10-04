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

def gpio_execute(operation, pin, level=None):
    operation = operation.strip().upper()

    try:
        if operation == "READ":
            value = hardware.gpio_read(pin)
            message = f"GPIO {pin} is {'HIGH' if value else 'LOW'}"

        elif operation == "SET":
            if level not in (0, 1):
                raise ValueError("SET requires level 0 or 1")
            value = hardware.gpio_set(pin, level)
            message = f"GPIO {pin} set to {'HIGH' if value else 'LOW'}"

        elif operation == "TOGGLE":
            old_value = hardware.gpio_read(pin)
            value = hardware.gpio_toggle(pin)
            message = (
                f"GPIO {pin} changed from "
                f"{'HIGH' if old_value else 'LOW'} to "
                f"{'HIGH' if value else 'LOW'}"
            )

        else:
            return {
                "success": False,
                "operation": operation,
                "pin": pin,
                "message": "GPIO operation not allowed"
            }

        return {
            "success": True,
            "operation": operation,
            "pin": pin,
            "value": value,
            "message": message
        }

    except Exception as e:
        return {
            "success": False,
            "operation": operation,
            "pin": pin,
            "message": str(e)
        }
