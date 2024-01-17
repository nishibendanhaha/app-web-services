import os
import sys
import cv2 as cv
import ffmpy


def image_to_video(img_list, output_path):
    weight = 128
    height = 256
    fps = 30
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    videowriter = cv.VideoWriter(output_path, fourcc, fps, (weight, height))
    for img in img_list:
        frame = cv.imread(img)
        videowriter.write(frame)
    videowriter.release()


def toh256(input_path, output_path):
    ff = ffmpy.FFmpeg(
        executable="/Users/huangtehui/Downloads/ffmpeg",
        inputs={input_path: None},
        outputs={output_path: '-c:v libx265'}
    )
    f = open(os.devnull, 'w')
    ff.run(stdout=f)
    f.close()
