# alphapose-extra

## Installation

```sh
pip install -r requirements.txt
```
## Run

Single picture
```sh
python image.py -i [image]
```

Multiple pictures
```sh
python image.py -d [directory]
```

Video
```sh
python video [input-video] [output-video]
```

Webcam test
```sh
python webcam.py --config webcam_cfg/test_camera.py
```

Webcam pose tracking
```sh
python webcam.py --config webcam_cfg/pose_tracking.py
```