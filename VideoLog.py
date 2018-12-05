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

# GLOBAL VARIABLES
PROJECT_DIR = dir_path = os.path.dirname(os.getcwd()) + "\\VideoLog\\"
model = None


class VideoLogModel:
    def __init__(self, project_dir):
        self.__model__ = ml.load_model(
            weight_path=project_dir + "saved_weights\\ai_eazyrank_rank_d169_nc_21_ver_4.0_tf.h5",
            num_classes=21)
        ml.ini_model(self.__model__)

    def pred_img(self, img, batch_size):
        pred = self.__model__.predict(img, batch_size=batch_size, verbose=1)
        return pred_utils.process_rank_prediction(pred)


def make_temp_dir(dir_name):
    tmp_dir = PROJECT_DIR + dir_name + "\\"

    # Create directory for decoding video
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    return tmp_dir


# originally passed
# sys.argv
def log_video(vid_dir):
    model = VideoLogModel(project_dir=PROJECT_DIR)
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(vid_dir)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    rec_rate = int(video_length % frame_rate / 2)
    print("Number of frames: ", video_length)
    print("Frame rate: " + str(frame_rate))
    print("Track rate: " + str(rec_rate))
    frame_num = 0

    frames = []
    frame_index = []
    frame_details = {}

    # Start converting the video
    while cap.isOpened():
        # Extract the frames as every half second interval
        ret, frame = cap.read()

        if (frame_num % rec_rate) == 0:
            t_frame = img_utils.square_wrap(frame)
            r_frame = cv2.resize(t_frame, (224, 224))
            frames.append(r_frame)
            frame_index.append(frame_num)

        frame_num = np.add(frame_num, 1)

        # If there are no more frames left
        if frame_num > np.subtract(video_length, 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("\nCompleted in %d seconds." % (time_end - time_start))
            break

    np_frame = np.asarray(frames)
    np_frame_num = np.asarray(frame_index)
    np_frame = np.reshape(a=np_frame, newshape=(-1, 224, 224, 3))

    print("\nNum of prediction images: " + str(len(np_frame)))
    print("Num of labels: " + str(len(np_frame_num)))

    pred = model.pred_img(img=np_frame, batch_size=24)

    # Output prediction results for later use
    for i in range(len(pred)):
        indx_data = []
        rank_data, output = pred_utils.process_rank_prediction(pred)
        indx_data.append(i)
        indx_data.append(rank_data)
        for _detail in output:
            indx_data.append(_detail)

        frame_details[str(i)] = {"rank": rank_data, "details": output}
    return frame_details


if __name__ == '__main__':
    # log_video(vid_dir=PROJECT_DIR + "logs\\test.mp4")
    m = u"・ メッセージ" + u"\n"
    m.encode("UTF-8")
    print(m)