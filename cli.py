import typer
from image import run_image
from video import run_video

app = typer.Typer()


@app.command()
def image(path: str, batch: bool = False, device: str = "cpu"):
    run_image(path, batch, device)


@app.command()
def video(input_path: str, output_path: str, device: str = "cpu"):
    run_video(input_path, output_path, device)


if __name__ == "__main__":
    app()
