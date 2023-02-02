# alphapose-extra

![Based on `mmcv`](https://img.shields.io/badge/based%20on-mmcv-blueviolet?style=flat-square) ![Supported Python Versions](https://img.shields.io/pypi/pyversions/mmcv?style=flat-square) ![License](https://img.shields.io/github/license/91b4dd62/alphapose-extra?style=flat-square)

## Installation

```bash
$ conda create --name openmmlab python=3.8 -y
$ conda activate openmmlab
$ pip install -r requirements.txt
```

## Run locally

### Image

```sh
python cli.py image [image | directory] [--batch] [--output]
```
`--batch` for multiple pictures (all the pictures in the directory)  
`--output` for writing the json output to a file instead of printing to console.

### Video

```sh
python cli.py video [input-video] [output-video]
```
`--output` for writing the json output to a file. (very big)

### Webcam test

```sh
python cli.py webcam --config webcam_cfg/test_camera.py
```
Press Q to quit.

### Webcam pose tracking

```sh
python cli.py webcam --config webcam_cfg/pose_tracking.py
```
