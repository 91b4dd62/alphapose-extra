from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
import tempfile
import os.path as osp
from lib import MMPose
from tqdm import tqdm
import cv2
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

device = 'cpu'
model = MMPose(backbone='SCNet')
pose_model = init_pose_model(
    model.pose_config, model.pose_checkpoint, device='cpu')
det_model = init_detector(
    model.det_config, model.det_checkpoint, device='cpu')


def process_frame(frame):
    mmdet_results = inference_detector(det_model, frame)
    person_results = process_mmdet_results(mmdet_results, cat_id=1)
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                                   frame,
                                                                   person_results,
                                                                   bbox_thr=0.3,
                                                                   format='xyxy',
                                                                   dataset=pose_model.cfg.data.test.type)
    vis_result = vis_pose_result(pose_model,
                                 frame,
                                 pose_results,
                                 dataset=pose_model.cfg.data.test.type,
                                 show=False)
    return vis_result


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
