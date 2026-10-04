from gpiozero import LED

LED_PIN = 17

_led = LED(LED_PIN)

def led_on():
    _led.on()
    return "LED is ON"

def led_off():
    _led.off()
    return "LED is OFF"

def get_state():
    return {
        "led": "on" if _led.is_lit else "off",
        "led_gpio": LED_PIN
    }

def shutdown():
    _led.off()
    _led.close()

def gpio_read(pin):
    if pin != LED_PIN:
        raise ValueError(f"GPIO {pin} is not authorized")
    return 1 if _led.is_lit else 0

def gpio_set(pin, level):
    if pin != LED_PIN:
        raise ValueError(f"GPIO {pin} is not authorized")

    if level not in (0, 1):
        raise ValueError("GPIO level must be 0 or 1")

    if level == 1:
        _led.on()
    else:
        _led.off()

    return gpio_read(pin)

def gpio_toggle(pin):
    if pin != LED_PIN:
        raise ValueError(f"GPIO {pin} is not authorized")

    _led.toggle()
    return gpio_read(pin)
