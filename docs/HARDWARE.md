# NeuroBrain Hardware

## Current
- Raspberry Pi 4, 4 GB RAM.
- Logitech C270 HD Webcam: USB microphone + camera.
- Speaker/audio output.
- LED on BCM GPIO17 (physical pin 11 in the tested setup).

## GPIO
`hardware.py` defines `LED_PIN = 17`.
Safe operations: READ, SET 0/1, TOGGLE.
Unauthorized pins are rejected.

Manual diagnostic:
```bash
pinctrl set 17 op dh
pinctrl set 17 op dl
```

## C270 video
Working capture device: `/dev/video0`.
Runtime uses MJPG 640×480 @ 30 FPS through OpenCV/V4L2.

## C270 microphone
Known working ALSA test:
```bash
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 5 /home/drpeker/test.wav
aplay /home/drpeker/test.wav
```
Python voice loop currently uses sounddevice device 1, mono 16 kHz.

## Future
Use ToF/ultrasonic for real collision distance. Bounding-box area is not a range sensor.
Add IMU and wheel encoders for orientation and odometry.
