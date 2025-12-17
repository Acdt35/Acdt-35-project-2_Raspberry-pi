from picamera2 import Picamera2   # required for camera module v3
import numpy as np


class MyPiCamera():

    def __init__(self, width, height):
        self.cap = Picamera2()

        self.width = width
        self.height = height
        self.is_open = True

        try:
            self.config = self.cap.create_video_configuration(
                main={"format": "RGB888", "size": (width, height)}
            )
            self.cap.align_configuration(self.config)
            self.cap.configure(self.config)
            self.cap.start()

        except:
            self.is_open = False
            return

    def read(self, dst=None):
        # allocate blank image to avoid returning "None"
        if dst is None:
            dst = np.empty((self.height, self.width, 3), dtype=np.uint8)

        if self.is_open:
            dst = self.cap.capture_array()

        # dst is either blank or the captured frame
        return self.is_open, dst

    def isOpened(self):
        return self.is_open

    def release(self):
        if self.is_open is True:
            self.cap.close()
        self.is_open = False
        return


if __name__ == "__main__":
    import cv2

    camera = MyPiCamera(640, 480)

    while camera.isOpened():
        _, image = camera.read()
        cv2.imshow("mycamera", image)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()