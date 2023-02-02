from lib import MMPose
from tqdm import tqdm
import json
import cv2
import os


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


def run_image(path: str, batch: bool, device: str, output: bool):
    model = MMPose(backbone='SCNet')
    if not batch:
        if not (os.path.isfile(path)):
            raise Exception("File not found")
        if not (path.endswith(".jpg") or path.endswith(".png")):
            raise Exception("File not supported")
        result = model.inference(img=path, device=device,
                                 show=True, name=os.path.splitext(path)[0]+"_output")
        final = json.dumps({path: processResult(result)}, indent=4)
        if output:
            with open(os.path.dirname(path)+'/output.json', 'w') as json_file:
                json_file.write(final)
        else:
            print(final)
    else:
        if not os.path.isdir(path):
            raise Exception("Directory not found")
        total = len([name for name in os.listdir(path)
                     if (name.endswith(".jpg") or name.endswith(".png")) and name.find("_output") < 0])

        from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                                 vis_pose_result, process_mmdet_results)
        from mmdet.apis import inference_detector, init_detector
        import tempfile
        import os.path as osp

        pose_model = init_pose_model(
            model.pose_config, model.pose_checkpoint, device=device)
        det_model = init_detector(
            model.det_config, model.det_checkpoint, device=device)

        bar = tqdm(total=total)
        result = {}

        for file in os.listdir(path):
            if (file.endswith(".jpg") or file.endswith(".png")) and not file.endswith("_output.png"):
                img = osp.join(path, file)

                # Rewrite Inference function
                mmdet_results = inference_detector(det_model, img)
                person_results = process_mmdet_results(
                    mmdet_results, cat_id=1)
                pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                                               img,
                                                                               person_results,
                                                                               bbox_thr=0.3,
                                                                               format='xyxy',
                                                                               dataset=pose_model.cfg.data.test.type)
                vis_result = vis_pose_result(pose_model,
                                             img,
                                             pose_results,
                                             dataset=pose_model.cfg.data.test.type,
                                             show=False)

                result[file] = processResult(pose_results)

                with tempfile.TemporaryDirectory() as tmpdir:
                    file_name = os.path.splitext(img)[0]+'_output.png'
                    cv2.imwrite(file_name, vis_result)
                    file_name1 = osp.join(path+'/static/images/', 'test.jpg')
                    cv2.imwrite(file_name1, vis_result)

                bar.update()
        bar.close()
        if output:
            with open(path+'/output.json', 'w') as json_file:
                json_file.write(json.dumps(result))
        else:
            print(final)
