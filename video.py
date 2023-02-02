from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from lib import MMPose
from tqdm import tqdm
import json
import cv2
import os

cache_model = None
cache_pose_model = None
cache_det_model = None


def processResult(result: list):
    final = []
    for i in result:
        score = 0
        keypoints = []
        for j in i['keypoints'].tolist():
            for k in j:
                keypoints.append(k)
        final.append({'bbox': i['bbox'].tolist(),
                      'score': score, 'keypoints': keypoints})
    return final


def initialize_models(device):
    global cache_model, cache_pose_model, cache_det_model
    if cache_model == None:
        cache_model = MMPose(backbone='SCNet')
    if cache_pose_model == None:
        cache_pose_model = init_pose_model(
            cache_model.pose_config, cache_model.pose_checkpoint, device=device)
    if cache_det_model == None:
        cache_det_model = init_detector(
            cache_model.det_config, cache_model.det_checkpoint, device=device)


def process_frame(frame):
    mmdet_results = inference_detector(cache_det_model, frame)
    person_results = process_mmdet_results(mmdet_results, cat_id=1)
    pose_results, returned_outputs = inference_top_down_pose_model(cache_pose_model,
                                                                   frame,
                                                                   person_results,
                                                                   bbox_thr=0.3,
                                                                   format='xyxy',
                                                                   dataset=cache_pose_model.cfg.data.test.type)
    vis_result = vis_pose_result(cache_pose_model,
                                 frame,
                                 pose_results,
                                 dataset=cache_pose_model.cfg.data.test.type,
                                 show=False)
    return (pose_results, vis_result)


def run_video(input_file, output_file, device, output):
    if not os.path.isfile(input_file):
        raise Exception("File not found")
    if not (input_file.endswith(".mp4")):
        raise Exception("File not supported")
    initialize_models(device)

    video_capture = cv2.VideoCapture(input_file)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    video_writer = None
    result = {}

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
            video_writer.write(proc[1])
            result[input_file] = processResult(proc[0])
            bar.update()
        else:
            print("Could not read the frame.")
            break

    video_capture.release()

    if video_writer != None:
        video_writer.release()

    bar.close()
    if output:
        with open(os.path.splitext(input_file)[0] + '_output.json', 'w') as json_file:
            json_file.write(json.dumps(result))
    else:
        print(result)
