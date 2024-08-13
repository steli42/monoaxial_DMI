#!/usr/bin/bash
python plots/spin_expval_series.py
python plots/make_video_txt_file.py
ffmpeg -r 24 -f concat -i video.txt out.mp4
ffmpeg -r 24 -f concat -i video.txt -s 320x320 out.gif