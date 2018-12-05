import os
import time

import cv2

PROJECT_DIR = dir_path = os.path.dirname(os.getcwd())


def make_temp_dir(dir_name):
    tmp_dir = PROJECT_DIR + "\\tmp\\" + dir_name + "\\"

    try:
        # Create directory for decoding video
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

    except OSError as e:
        print("Error creating file dir for videodata...")
        print(e.errno)

    return tmp_dir


def decode_video(vid_dir, output_dir):
    tmp_dir = make_temp_dir(dir_name=output_dir)
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
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if count % 5 == 0:
            cv2.imwrite(tmp_dir + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % (count / 10))
            print("Completed in %d seconds." % (time_end - time_start))
            break

    print("Video decoded to: " + tmp_dir)
    return tmp_dir


if __name__ == "__main__":
    # print(str(make_temp_dir("vine")))
    decode_video(vid_dir="C:\\Users\\Jwaltonp\\Desktop\\Pics\\input\\6.mp4", output_dir="Human")

