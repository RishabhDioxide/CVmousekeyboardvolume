# CVmousekeyboardvolume
A virtual Mouse and Keyboard with Volume Control and Mode switching in real time using Computer Vision. (windows only)
Created and Tested on SPYDER using Python 3.10.

Mandatory Libs: opencv as cv2, mediapipe, time, math, pynput.keyboard, autopy, pycaw, numpy

What the system does - 
- Virtual Mouse: Cursor movement using index-finger tracking, left/right clicks via finger-touch gestures
- Virtual Keyboard: On-screen QWERTY keyboard with gesture-based key selection and typing
- Volume Control: System audio adjusted using thumb–index finger distance.
All three modules were developed and tested independently, and later merged into a single system.
One part I spent a lot of time on was real-time mode switching without any physical input.
If the pinky finger hovers over one of the on-screen mode buttons (Mouse / Keyboard / Volume) for a short duration, the system switches to that mode automatically. This keeps the interaction fully gesture-driven and avoids relying on keyboard shortcuts or manual toggles.

Technical stack: Python, OpenCV, MediaPipe, Autopy, Pynput, Pycaw
This project aligns with my interest in Computer Vision, Human–Computer Interaction, and real-time systems, and I’m continuing to build more work in this space.
