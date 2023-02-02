from lib import MMPose
from tqdm import tqdm
import cv2
import os


def run_image(path: str, batch: bool, device: str):
    model = MMPose(backbone='SCNet')
    if not batch:
        result = model.inference(img=path, device=device,
                                 show=True, name=os.path.splitext(path)[0]+"_output")
        print(result)
    else:
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

        for file in os.listdir(path):
            if file.endswith(".jpg") or file.endswith(".png"):
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

                with tempfile.TemporaryDirectory() as tmpdir:
                    file_name = os.path.splitext(img)[0]+'_output.png'
                    cv2.imwrite(file_name, vis_result)
                    file_name1 = osp.join(path+'/static/images/', 'test.jpg')
                    cv2.imwrite(file_name1, vis_result)

                bar.update()
