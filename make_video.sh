#!/usr/bin/bash
python launch_jobs/plots/spin_expval_series.py
python launch_jobs/plots/make_video_txt_file.py
ffmpeg -r 24 -f concat -i video.txt out.mp4