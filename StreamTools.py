from __future__ import print_function
import sys
import cv2
import os
import numpy as np
from utils import model_loader as ml, labels
from utils import img_utils, pred_utils
import threading
import multiprocessing
import time


class StreamModel:
    def __init__(self, project_dir):
        self.__model__ = ml.load_model(weight_path=project_dir + "saved_weights\\ai_eazyrank_rank_d169_nc_21_ver_4.0_tf.h5",
                                       num_classes=21)
        ml.ini_model(self.__model__)

    def pred_img(self, img):
        pred_img = img_utils.prep_pred_img(img, 224, 224)
        pred = self.__model__.predict(pred_img, batch_size=1, verbose=1)
        return pred_utils.process_rank_prediction(pred)


PROJECT_DIR = dir_path = os.path.dirname(os.getcwd()) + "\\VideoLog\\"
model = None


def make_temp_dir(dir_name):
    tmp_dir = PROJECT_DIR + dir_name + "\\"

    # Create directory for decoding video
    if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
    return tmp_dir


# originally passed
# sys.argv
def main(save_dir):
    """
    Change the camera setting using the set() function

       0  CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
       1  CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
       2  CAP_PROP_POS_AVI_RATIO Relative position of the video file
       3  CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
       4  CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
       5  CAP_PROP_FPS Frame rate.
       6  CAP_PROP_FOURCC 4-character code of codec.
       7  CAP_PROP_FRAME_COUNT Number of frames in the video file.
       8  CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
       9 CAP_PROP_MODE Backend-specific value indicating the current capture mode.
       10 CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
       11 CAP_PROP_CONTRAST Contrast of the image (only for cameras).
       12 CAP_PROP_SATURATION Saturation of the image (only for cameras).
       13 CAP_PROP_HUE Hue of the image (only for cameras).
       14 CAP_PROP_GAIN Gain of the image (only for cameras).
       15 CAP_PROP_EXPOSURE Exposure (only for cameras).
       16 CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
       17 CAP_PROP_WHITE_BALANCE Currently unsupported
       18 CAP_PROP_RECTIFICATION Rectification flag for stereo camera


    :param argv:
    :return:
        Read the current setting from the camera
    """

    # capture from camera at location 0
    cap = cv2.VideoCapture(0)

    # Set camera settings
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
    cap.set(cv2.CAP_PROP_GAIN, 4.0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 144.0)
    cap.set(cv2.CAP_PROP_CONTRAST, 27.0)
    cap.set(cv2.CAP_PROP_HUE, 13.0)  # 13.0
    cap.set(cv2.CAP_PROP_SATURATION, 28.0)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    test = cap.get(cv2.CAP_PROP_POS_MSEC)
    ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    hue = cap.get(cv2.CAP_PROP_HUE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    exposure = cap.get(cv2.CAP_PROP_SATURATION)

    print("Test: ", test)
    print("Ratio: ", ratio)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    print("Brightness: ", brightness)
    print("Contrast: ", contrast)
    print("Saturation: ", saturation)
    print("Hue: ", hue)
    print("Gain: ", gain)
    print("Exposure: ", exposure)

    i = 0
    sec = 0
    while True:
        ret, img = cap.read()
        cv2.imshow("Input", img)

        if(i % 15) == 0:
            if save_dir and len(save_dir) > 0:
                # cv2.imwrite(save_dir + "frame_" + str(sec) + ".jpg", img)
                r, o = stream_model.pred_img(img)
                img_details = ""

                # 日本語のレスポンスを設定する
                if "None" in o.get("img_details"):
                    img_details = "None"
                else:
                    index = 0
                    for i in range(len(o.get("img_details"))):
                        # Append all details to image
                        img_details = img_details + o.get("img_details")[i]
                        if index < (len(o.get("img_details")) - 1):
                            img_details = img_details + ", "
                        index = index + 1

                print("Img_rank: " + str(o.get("rank")))
                print("Confidence: " + str(o.get("confidence")))
                print("Img_details: " + str(img_details))

            sec = np.add(sec, 1)

        key = cv2.waitKey(10)
        if key == 27:
            break
        i = np.add(i, 1)

    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()


if __name__ == '__main__':
    m = multiprocessing.Manager()
    q = m.Queue()

    stream_model = StreamModel(project_dir=PROJECT_DIR)

    td = make_temp_dir("ok")
    main(td)
