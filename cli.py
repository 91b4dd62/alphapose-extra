import typer
import torch

from image import run_image
from video import run_video
from webcam import run_webcam
from basic_recognition import display_pose

app = typer.Typer()


def auto_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


@app.command()
def image(path: str, batch: bool = False, device: str = auto_device(), output: bool = False):
    run_image(path, batch, device, output)


@app.command()
def video(input_path: str, output_path: str, device: str = auto_device(), output: bool = False, poseflow: bool = False):
    run_video(input_path, output_path, device=device, output=output, poseflow=poseflow)


@app.command()
def webcam(config: str = "webcam_cfg/pose_tracking.py", device: str = auto_device()):
    run_webcam(config=config, device=device)


@app.command()
def recognition(path: str, outpath: str, tracked: str, limit: float = 0.03):
    display_pose(path, outpath, tracked, limit=limit)

if __name__ == "__main__":
    app()
