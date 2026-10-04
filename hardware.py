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
