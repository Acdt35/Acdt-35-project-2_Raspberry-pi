import sys
sys.path.append("/home/acdt35/AI_CAR")

import mycamera
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from gpiozero import DigitalOutputDevice, PWMOutputDevice

import time
import onnxruntime as ort

TL_MODEL_PATH  = "/home/acdt35/AI_CAR/model/traffic_light_best.onnx"
TL_IMG_SIZE    = 416
TL_CONF_THRES  = 0.60

PRINT_INTERVAL   = 5.0
COLOR_MIN_PIXELS = 100
MIN_TL_BOX_H     = 70  # increase if detect too early

TL_INFER_INTERVAL = 0.2
TL_HOLD_TIME      = 0.80

GHOST_DROP_STREAK = 4

cv2.setNumThreads(1)


PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

def motor_go(speed):
    AIN1.value = 0; AIN2.value = 1; PWMA.value = speed
    BIN1.value = 0; BIN2.value = 1; PWMB.value = speed

def motor_left(speed):
    AIN1.value = 1; AIN2.value = 0; PWMA.value = 0.0
    BIN1.value = 0; BIN2.value = 1; PWMB.value = speed

def motor_right(speed):
    AIN1.value = 0; AIN2.value = 1; PWMA.value = speed
    BIN1.value = 1; BIN2.value = 0; PWMB.value = 0.0

def motor_stop():
    PWMA.value = 0.0
    PWMB.value = 0.0


speedSet = 0.7
speedSet_slow = 0.4

def img_preprocess(image):
    h, _, _ = image.shape
    image = image[int(h/2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    return image / 255.0

def letterbox(im, new_shape=416, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(w * r), int(h * r)

    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    pw, ph = (new_shape - nw) // 2, (new_shape - nh) // 2

    im_padded = cv2.copyMakeBorder(
        im_resized, ph, ph, pw, pw,
        cv2.BORDER_CONSTANT, value=color
    )
    return im_padded, r, (pw, ph)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def scale_coords(boxes, r, pad, shape):
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= r

    h, w = shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w-1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h-1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w-1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h-1)
    return boxes

def tl_preprocess(frame):
    img, r, pad = letterbox(frame, TL_IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img[np.newaxis], r, pad

def tl_postprocess(pred, r, pad, shape):
    pred = pred[0]

    if pred.ndim != 2 or pred.shape[0] == 0 or pred.shape[1] < 6:
        return []

    boxes = pred[:, :4]
    obj   = pred[:, 4:5]
    cls   = pred[:, 5:]

    cls_id = np.argmax(cls, axis=1)
    cls_score = cls[np.arange(len(cls)), cls_id].reshape(-1, 1)

    conf = (obj * cls_score).reshape(-1)

    keep = conf >= TL_CONF_THRES
    boxes = boxes[keep]

    if len(boxes) == 0:
        return []

    boxes = xywh2xyxy(boxes)
    boxes = scale_coords(boxes, r, pad, shape)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    return [boxes[best_idx]]

def classify_traffic_light_color(frame, box):
    x1, y1, x2, y2 = box.astype(int)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 80), (179, 255, 255))
    red_mask = red1 | red2

    yellow_mask = cv2.inRange(hsv, (18, 80, 80), (35, 255, 255))
    green_mask  = cv2.inRange(hsv, (36, 80, 80), (85, 255, 255))

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "yellow": int(cv2.countNonZero(yellow_mask)),
        "green": int(cv2.countNonZero(green_mask)),
    }

    best = max(counts, key=counts.get)
    if counts[best] < COLOR_MIN_PIXELS:
        return "none"
    return best

def draw_tl_box(frame, box):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "traffic light", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def main():
    camera = mycamera.MyPiCamera(640, 480)
    lane_model = load_model("/home/acdt35/AI_CAR/model/lane_navigation_final.keras")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    tl_sess = ort.InferenceSession(
        TL_MODEL_PATH,
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )
    tl_in  = tl_sess.get_inputs()[0].name
    tl_out = tl_sess.get_outputs()[0].name

    carState = "stop"
    traffic_light_state = "none"

    last_print_time = 0.0

    last_tl_infer_time = 0.0
    last_tl_seen_time  = 0.0
    last_box = None

    ghost_streak = 0

    pre = np.zeros((66, 200, 3), dtype=np.float32)

    while camera.isOpened():
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 82:
            carState = "go"
        elif key == 84:
            carState = "stop"

        _, frame = camera.read()
        frame = cv2.flip(frame, -1)

        # lane prediction
        pre = img_preprocess(frame)
        angle = int(lane_model.predict(np.asarray([pre]), verbose=0)[0])

        now = time.time()

        do_tl_infer = (now - last_tl_infer_time) >= TL_INFER_INTERVAL
        box_to_use = None

        if do_tl_infer:
            last_tl_infer_time = now

            tl_inp, r, pad = tl_preprocess(frame)
            pred = tl_sess.run([tl_out], {tl_in: tl_inp})[0]
            boxes = tl_postprocess(pred, r, pad, frame.shape)

            if len(boxes) > 0:
                box = boxes[0]
                if (box[3] - box[1]) >= MIN_TL_BOX_H:
                    box_to_use = box
                    last_box = box
                    last_tl_seen_time = now
                else:
                    box_to_use = None
        else:
            if last_box is not None and (now - last_tl_seen_time) <= TL_HOLD_TIME:
                box_to_use = last_box

        traffic_light_state = "none"
        if box_to_use is not None:
            traffic_light_state = classify_traffic_light_color(frame, box_to_use)

            if traffic_light_state in ("none", "unknown"):
                ghost_streak += 1
            else:
                ghost_streak = 0

            if ghost_streak >= GHOST_DROP_STREAK:
                last_box = None
                box_to_use = None
                traffic_light_state = "none"
                ghost_streak = 0
        else:
            ghost_streak = 0

        if now - last_print_time >= PRINT_INTERVAL:
            print("Traffic light color:", traffic_light_state)
            last_print_time = now

        vis = frame.copy()
        if box_to_use is not None:
            vis = draw_tl_box(vis, box_to_use)

        cv2.imshow("Original", vis)
        cv2.imshow("pre", pre)

        if carState == "go":
            if traffic_light_state == "red":
                motor_stop()
                continue

            cur_speed = speedSet_slow if traffic_light_state == "yellow" else speedSet

            if 85 <= angle <= 95:
                motor_go(cur_speed)
            elif angle > 96:
                motor_right(cur_speed)
            else:
                motor_left(cur_speed)
        else:
            motor_stop()

    motor_stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()