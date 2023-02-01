from lib import MMPose
from tqdm import tqdm
import cv2
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

model = MMPose(backbone='SCNet')


def process_frame(frame):
    result = model.inference(img=frame, device='cpu',
                             save=False, show=False, return_vis_result=True)
    return result


def main():
    video_capture = cv2.VideoCapture(input_file)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    video_writer = None

    bar = tqdm(total=total_frames)

    while video_capture.isOpened():
        frame_is_read, frame = video_capture.read()

        if video_writer is None:
            video_writer = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*"avc1"),
                target_fps,
                (frame.shape[1], frame.shape[0]),
                True,
            )

        if frame_is_read and frame is not None:
            proc = process_frame(frame)
            video_writer.write(proc)
            bar.update()
        else:
            print("Could not read the frame.")
            break

    video_capture.release()

    if video_writer != None:
        video_writer.release()


if __name__ == "__main__":
    main()
