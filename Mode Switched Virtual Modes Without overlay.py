# virtual_control_app.py
# Main application file for gesture based control
# Written during experimentation, so some values are tuned by trial & error

import cv2 as cv
import mediapipe as mp
import time
import math
from time import sleep
from pynput.keyboard import Controller as PynputController
import autopy
# For Volume Control Bridge b/w OS and Python (windows specific)
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pynput import keyboard as pk
import threading
import numpy as np

MODES = ["MOUSE", "KEYBOARD", "VOLUME"]


class VirtualKeyboard:
    def __init__(self):
        # camera geometry (not used here but kept for parity)
        self.finaltext = ""
        self.keyboard = PynputController()
        # keyboard layout (simple QWERTY)
        self.button_size = (70, 70)
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', ' ', '=']
        ]
        # store button positions
        self.button_list = []
        for i in range(len(self.keys)):
            for j, key in enumerate(self.keys[i]):
                # spacing chosen after some manual testing
                x = 100 * j + 30
                y = 100 * i + 60
                self.button_list.append({'pos': (x, y), 'text': key, 'size': self.button_size})
                
        # simple debounce so key doesn't repeat too fast
        self.last_type = 0.0
        self.type_delay = 0.35  # seconds (might feel slow, but avoids spam)

    def draw_buttons(self, frame):
        for b in self.button_list:
            x, y = b['pos']
            w, h = b['size']
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), cv.FILLED)
            cv.putText(frame, b['text'], (x + 13, y + 38),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
        # show typed text box, text preview box
        cv.rectangle(frame, (10, 400), (700, 450), (175, 0, 175), cv.FILLED)
        cv.putText(frame, self.finaltext, (20, 435), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, "Virtual Keyboard", (350, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    def process(self, frame, results, fps):
        """
        frame: BGR numpy array (already mirrored)
        results: mediapipe hand detection results
        fps: current fps value (for overlay)
        """
        lmList = []
        if results and results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            h, w, c = frame.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mp.solutions.drawing_utils.draw_landmarks(frame, myHand, mp.solutions.hands.HAND_CONNECTIONS)

        # default overlay
        self.draw_buttons(frame)
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

        if len(lmList) != 0:
            # index tip, middle tip
            x1, y1 = lmList[8][1], lmList[8][2]
            x2, y2 = lmList[12][1], lmList[12][2]
            # visualize fingertips
            cv.circle(frame, (x1, y1), 5, (255, 0, 0), cv.FILLED)
            cv.circle(frame, (x2, y2), 5, (255, 0, 0), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

            # hover / press detection
            for b in self.button_list:
                x, y = b['pos']
                w, h = b['size']
                if x < x1 < x + w and y < y1 < y + h:
                    if length >= 31: # hard-coded threshold (could be tuned better)
                        cv.rectangle(frame, (x, y), (x + w, y + h), (175, 0, 175), cv.FILLED)
                    else:
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), cv.FILLED)
                        now = time.time()
                        if now - self.last_type > self.type_delay:
                            # type the character
                            ch = b['text']
                            # handle space specially
                            if ch == ' ':
                                self.keyboard.press(' ')
                                sleep(0.01)
                                self.keyboard.release(' ')
                                self.finaltext += ' '
                            else:
                                # press and release
                                self.keyboard.press(ch)
                                sleep(0.01)
                                self.keyboard.release(ch)
                                self.finaltext += ch
                            self.last_type = now
                    cv.putText(frame, b['text'], (x + 13, y + 38), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)

        return frame


class VirtualMouse:
    def __init__(self, widthcam=960, heightcam=600):
        self.widthcam = widthcam
        self.heightcam = heightcam
        self.box_w, self.box_h = 800, 500
        self.x_margin = (widthcam - self.box_w) // 2
        self.y_margin = 5
        self.tipIds = [4, 8, 12, 16, 20]
        self.prev_state = None
        self.wscr, self.hscr = autopy.screen.size()
        # smoothing
        self.smoothening = 3 # tuned after testing
        self.prevlocationX, self.prevlocationY = 0, 0

    def fingers_up(self, lmList):
        fingers = []
        # check thumb - comparing x coords (works normally when palm faces camera)
        # thumb logic is not perfect but works for front-facing hand
        try:
            if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        except Exception:
            fingers.append(0)

        for id in range(1, 5):
            try:
                if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            except Exception:
                fingers.append(0)
        return fingers

    def process(self, frame, results, fps):
        lmListmain = []
        if results and results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            h, w, c = frame.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmListmain.append([id, cx, cy])
            mp.solutions.drawing_utils.draw_landmarks(frame, myHand, mp.solutions.hands.HAND_CONNECTIONS)

        if len(lmListmain) != 0:
            x1, y1 = lmListmain[8][1], lmListmain[8][2]  # index tip
            x2, y2 = lmListmain[12][1], lmListmain[12][2]  # middle tip
            xi, yi = lmListmain[6][1], lmListmain[6][2]  # some intermediate point
            fingers = self.fingers_up(lmListmain)

            # Cursor mode: index up, middle down
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (self.x_margin, self.x_margin + self.box_w), (0, self.wscr))
                y3 = np.interp(y1, (self.y_margin, self.y_margin + self.box_h), (0, self.hscr))
                # Smoothening
                currlocationX = self.prevlocationX + (x3 - self.prevlocationX) / self.smoothening
                currlocationY = self.prevlocationY + (y3 - self.prevlocationY) / self.smoothening
                currlocationX = np.clip(currlocationX, 0, self.wscr - 1)
                currlocationY = np.clip(currlocationY, 0, self.hscr - 1)
                # Move mouse
                autopy.mouse.move(currlocationX, currlocationY)
                # Update previous location
                self.prevlocationX, self.prevlocationY = currlocationX, currlocationY
                cv.circle(frame, (x1, y1), 15, (255, 0, 0), cv.FILLED)
                cv.putText(frame, "Cursor Mode", (5, 60), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            # Clicking mode: index and middle both up
            if fingers[1] == 1 and fingers[2] == 1:
                length = math.hypot(x2 - x1, y2 - y1)
                lengthumb = math.hypot(xi - lmListmain[4][1], yi - lmListmain[4][2]) if len(lmListmain) > 4 else 999
                cv.putText(frame, "Clicking Mode", (5, 60), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                if length < 35:
                    autopy.mouse.click(button=autopy.mouse.Button.LEFT, delay=0.2)
                    cv.putText(frame, "CLICK !", (x2 - 30, y2 - 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                if lengthumb < 35:
                    autopy.mouse.click(button=autopy.mouse.Button.RIGHT, delay=0.2)
                    cv.putText(frame, "RIGHT CLICK !", (xi - 30, yi - 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            self.prev_state = "hand"
        else:
            if self.prev_state != "empty":
                print("List Empty")
                self.prev_state = "empty"

        # draw bounding box region
        cv.rectangle(frame, (self.x_margin, self.y_margin),
                     (self.x_margin + self.box_w, self.y_margin + self.box_h),
                     (0, 255, 0), 2)
        cv.putText(frame, "Virtual Mouse", (350, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

        return frame


class VolumeControl:
    def __init__(self):
        # initialize pycaw audio endpoint
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = interface.QueryInterface(IAudioEndpointVolume)
        # volume scalar range 0.0 .. 1.0
        self.minVol = 0.0
        self.maxVol = 1.0
        self.volBar = 400
        self.volpercent = 0
        # mediapipe drawing tool kept external

    def find_positions(self, results, frame, draw=False):
        lmList = []
        if results and results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            h, w, c = frame.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 7, (255, 0, 0), cv.FILLED)
            mp.solutions.drawing_utils.draw_landmarks(frame, myHand, mp.solutions.hands.HAND_CONNECTIONS)
        return lmList

    def process(self, frame, results, fps):
        # Draw UI
        cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
        cv.putText(frame, "Volume Control", (350, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

        lmList = self.find_positions(results, frame, draw=False)
        if len(lmList) != 0:
            # thumb tip id 4, index tip id 8
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cv.circle(frame, (x1, y1), 8, (225, 0, 225), cv.FILLED)
            cv.circle(frame, (x2, y2), 8, (225, 0, 225), cv.FILLED)
            cv.line(frame, (x1, y1), (x2, y2), (225, 0, 225), 2)
            linelength = math.hypot(x2 - x1, y2 - y1)
            # map length to volume
            myvolume = np.interp(linelength, [30, 260], [self.minVol, self.maxVol])
            self.volBar = np.interp(linelength, [30, 260], [400, 150])
            self.volpercent = np.interp(linelength, [30, 260], [0, 100])
            # set system volume
            self.volume.SetMasterVolumeLevelScalar(float(myvolume), None)
            # draw filled bar
            cv.rectangle(frame, (50, int(self.volBar)), (85, 400), (0, 255, 0), cv.FILLED)
            cv.putText(frame, f"{int(self.volpercent)} %", (40, 450), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        return frame


class VirtualControlApp:
    def __init__(self, cam_index=0):
        # single webcam
        self.cap = cv.VideoCapture(cam_index)
        self.widthcam, self.heightcam = 960, 600
        self.cap.set(3, self.widthcam)
        self.cap.set(4, self.heightcam)

        # Mediapipe Hands (single shared detector)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # modules
        self.mouse = VirtualMouse(widthcam=self.widthcam, heightcam=self.heightcam)
        self.keyboard = VirtualKeyboard()
        self.volume = VolumeControl()

        # mode and concurrency
        self.mode_index = 0  # start with MOUSE
        self.mode_lock = threading.Lock()

        # now store only names; positions computed each frame to avoid overlap
        self.mode_button_names = ["MOUSE", "KEYBOARD", "VOLUME"]
        self.hover_counts = {name: 0 for name in self.mode_button_names}
        self.HOLD_FRAMES = 20   # ~0.6 sec if ~30fps
        self.last_mode_switch = time.time()

        # hotkey listener (GlobalHotKeys)
        self.listener = None
        self._start_hotkey_listener()

        # FPS
        self.prev_time = 0

    def _start_hotkey_listener(self):
        """
        Use GlobalHotKeys for robust combination handling
        """
        from pynput import keyboard as _pk

        def toggle():
            self.toggle_next_mode()
            print("Hotkey: toggle ->", self.current_mode())

        def pick1():
            self.set_mode_index(0)
            print("Hotkey: set mode ->", self.current_mode())

        def pick2():
            self.set_mode_index(1)
            print("Hotkey: set mode ->", self.current_mode())

        def pick3():
            self.set_mode_index(2)
            print("Hotkey: set mode ->", self.current_mode())

        hotkey_map = {
            '<ctrl>+<alt>+x': toggle,
            '<ctrl>+<alt>+1': pick1,
            '<ctrl>+<alt>+2': pick2,
            '<ctrl>+<alt>+3': pick3,
        }

        self.listener = _pk.GlobalHotKeys(hotkey_map)
        self.listener.start()
        print("Global hotkeys started: Ctrl+Alt+X to toggle, Ctrl+Alt+1/2/3 to pick.")

    def set_mode_index(self, i):
        with self.mode_lock:
            self.mode_index = int(i) % len(MODES)
        print("Mode set to:", MODES[self.mode_index])

    def toggle_next_mode(self):
        with self.mode_lock:
            self.mode_index = (self.mode_index + 1) % len(MODES)
        print("Toggled mode to:", MODES[self.mode_index])

    def current_mode(self):
        with self.mode_lock:
            return MODES[self.mode_index]

    #hover helpers (bottom-right placement)
    def compute_button_rects(self, frame):
        """
        Return list of dicts: {'name':..., 'rect':(x,y,w,h)}
        Buttons are stacked vertically at extreme bottom-right with margins.
        """
        h, w = frame.shape[:2]
        btn_w, btn_h = 140, 50     # button size (wider, shorter)
        spacing = 8
        margin_right = 10
        margin_bottom = 10

        rects = []
        # stack from bottom upward
        for i, name in enumerate(self.mode_button_names):
            # y position of the bottom-most button = h - margin_bottom - btn_h
            y_bottom = h - margin_bottom - (i * (btn_h + spacing)) - btn_h
            # compute top-left corner
            x = w - margin_right - btn_w
            y = h - margin_bottom - (i * (btn_h + spacing)) - btn_h
            rects.append({'name': name, 'rect': (x, y, btn_w, btn_h)})
        return rects

    def draw_mode_buttons(self, frame):
        rects = self.compute_button_rects(frame)
        for r in rects:
            x, y, bw, bh = r['rect']
            # draw button background
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (50, 50, 50), cv.FILLED)
            # draw border
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 1)
            # draw label centered-ish
            text = r['name']
            cv.putText(frame, text, (x + 10, y + int(bh/2) + 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # progress indicator if hovering
            cnt = self.hover_counts.get(r['name'], 0)
            if cnt > 0:
                # draw a small progress bar at the bottom of the button
                progress_w = int((cnt / max(1, self.HOLD_FRAMES)) * bw)
                progress_w = max(2, min(bw, progress_w))
                cv.rectangle(frame, (x, y + bh - 6), (x + progress_w, y + bh - 2), (0, 200, 0), cv.FILLED)

    def check_hover_and_switch(self, lmList, frame):
        # no hand → reset hover counters
        if not lmList:
            for k in self.hover_counts:
                self.hover_counts[k] = 0
            return

        # find index finger tip landmark (ID = 8)
        tip = next((p for p in lmList if p[0] == 20), None)
        if tip is None:
            return

        ix, iy = tip[1], tip[2]
        rects = self.compute_button_rects(frame)

        for r in rects:
            name = r['name']
            x, y, w_btn, h_btn = r['rect']

            # is fingertip inside the button?
            if x < ix < x + w_btn and y < iy < y + h_btn:
                self.hover_counts[name] += 1

                # reset other counters
                for other in self.hover_counts:
                    if other != name:
                        self.hover_counts[other] = 0

                # enough frames held?
                if self.hover_counts[name] >= self.HOLD_FRAMES:
                    now = time.time()
                    # avoid rapid accidental switches
                    if now - self.last_mode_switch > 0.6:
                        self.set_mode_index(["MOUSE", "KEYBOARD", "VOLUME"].index(name))
                        print("Hover Switched →", name)
                        self.last_mode_switch = now

                    # reset counters after switching
                    for other in self.hover_counts:
                        self.hover_counts[other] = 0
                return
            else:
                self.hover_counts[name] = 0

#end hover helpers

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        print("Starting Virtual Control App. Default mode:", self.current_mode())
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
            self.prev_time = curr_time

            # route to active module
            mode = self.current_mode()
            if mode == "MOUSE":
                annotated = self.mouse.process(frame, results, fps)
            elif mode == "KEYBOARD":
                annotated = self.keyboard.process(frame, results, fps)
            elif mode == "VOLUME":
                annotated = self.volume.process(frame, results, fps)
            else:
                annotated = frame

            # on-screen hover switching
            # extract lmList (hand landmarks)
            lmList = []
            if results and results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                h, w, c = annotated.shape
                for id, lm in enumerate(hand.landmark):
                    lmList.append([id, int(lm.x * w), int(lm.y * h)])
            for p in lmList:
                if p[0] == 20:  # pinky tip
                    cv.circle(annotated, (p[1], p[2]), 10, (0, 255, 255), cv.FILLED)
            self.check_hover_and_switch(lmList, annotated)
            self.draw_mode_buttons(annotated)

            cv.imshow('Video', annotated)
            # quit with 'd' key
            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        self.cap.release()
        cv.destroyAllWindows()
        if self.listener:
            self.listener.stop()
        print("App terminated.")


if __name__ == "__main__":
    app = VirtualControlApp(cam_index=0)
    app.run()