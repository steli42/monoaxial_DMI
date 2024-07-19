#!/usr/bin/bash
python tdvp/plots/spin_expval_series.py
python tdvp/plots/make_video_txt_file.py
ffmpeg -r 48 -f concat -i video.txt out.mp4
ffmpeg -r 48 -f concat -i video.txt -s 320x320 out.gif