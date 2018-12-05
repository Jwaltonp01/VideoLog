import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import video_decoder as vd, model_loader, labels

if __name__ == "__main__":
    # vl_model = model_loader.load_new_model(old_weight_path="saved_weights\\rank_d169_nc_15_epch_8_ver_1_tf.h5",
    #                                        old_class_num=15,
    #                                        new_class_num=labels.get_num_labels(),
    #                                        lr=1e-03)
    # # # Decode the given video
    # output_dir = vd.decode_video(vid_dir="", output_dir="")
    # a = 0

    # print(vl_model.summary())
    frame_num = 1203
    frame_rate = 29
    rec_rate = frame_num % frame_rate/2
    print("Track rate: " + str(rec_rate))

    for i in range(frame_num):
        if i % rec_rate == 0:
            print("Frame recorded")
