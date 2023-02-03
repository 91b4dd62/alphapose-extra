# alphapose-extra

![Based on `mmcv`](https://img.shields.io/badge/based%20on-mmcv-blueviolet?style=flat-square) ![Supported Python Versions](https://img.shields.io/pypi/pyversions/mmcv?style=flat-square) ![License](https://img.shields.io/github/license/91b4dd62/alphapose-extra?style=flat-square)

https://media.githubusercontent.com/media/91b4dd62/alphapose-extra/main/examples/2.mp4

https://media.githubusercontent.com/media/91b4dd62/alphapose-extra/main/examples/2_output.mp4

## Installation

```bash
$ conda create --name openmmlab python=3.8 -y
$ conda activate openmmlab
$ pip install -r requirements.txt
```

## Run locally

### Image

```sh
python cli.py image <file or directory> [--batch] [--output]
```

- `--batch` for multiple pictures (all the pictures in the directory)  
- `--output` for writing the json output to a file instead of printing to console.

### Video

```sh
python cli.py video <input> <output> [--output] [--poseflow]
```

- `--output` for writing the json output to a file. (very big)
- `--poseflow` for generating each frame as well for [Poseflow](https://github.com/YuliangXiu/PoseFlow) support. **Poseflow isn't included in this repo.**  

Use the following command with Poseflow under the same directory to generate Poseflow trackers.

```sh
python tracker-general.py --imgdir [video_name]_frames 
                          --in_json [video_name]_output.json 
                          --out_json [video_name]_output_forvis_tracked.json 
                          --visdir [video_name]_tracked_frames
```

### Webcam test

```sh
python cli.py webcam --config webcam_cfg/test_camera.py
```

Press <kbd>Q</kbd> to quit.

### Webcam pose tracking

```sh
python cli.py webcam --config webcam_cfg/pose_tracking.py
```
