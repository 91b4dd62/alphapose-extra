import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# visualization (from poseflow)


def display_pose(imgdir, visdir, tracked, cmap):
    print("Start visualization...\n")
    for imgname in tqdm(tracked.keys()):
        img = Image.open(os.path.join(imgdir, imgname))
        width, height = img.size
        fig = plt.figure(figsize=(width/10, height/10), dpi=10)
        plt.imshow(img)
        tracked_id = 0
        for pid in range(len(tracked[imgname])):
            pose = np.array(tracked[imgname][pid]
                            ['keypoints']).reshape(-1, 3)[:, :3]
            tracked_id += 1

            # keypoint scores of torch version and pytorch version are different
            if np.mean(pose[:, 2]) < 1:
                alpha_ratio = 1.0
            else:
                alpha_ratio = 5.0

            coco_part_names = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow',
                               'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
            colors = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y',
                      'y', 'y', 'y', 'g', 'g', 'g', 'g', 'g', 'g']
            pairs = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [
                8, 10], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [6, 12], [5, 11]]
            for idx_c, color in enumerate(colors):
                plt.plot(np.clip(pose[idx_c, 0], 0, width), np.clip(pose[idx_c, 1], 0, height), marker='o',
                         color=color, ms=80/alpha_ratio*np.mean(pose[idx_c, 2]), markerfacecolor=(1, 1, 0, 0.7/alpha_ratio*pose[idx_c, 2]))
            for idx in range(len(pairs)):
                plt.plot(np.clip(pose[pairs[idx], 0], 0, width), np.clip(pose[pairs[idx], 1], 0, height), 'r-',
                         color=cmap(tracked_id), linewidth=60/alpha_ratio*np.mean(pose[pairs[idx], 2]), alpha=0.6/alpha_ratio*np.mean(pose[pairs[idx], 2]))
        plt.axis('off')
        ax = plt.gca()
        ax.set_xlim([0, width])
        ax.set_ylim([height, 0])
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if not os.path.exists(visdir):
            os.mkdir(visdir)
        fig.savefig(os.path.join(visdir, imgname.split(".")[
                    0]+".png"), pad_inches=0.0, bbox_inches=extent, dpi=13)
        plt.close()
        
def reader(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

display_pose('examples/','examples/vis/',reader('examples/output.json'),plt.cm.get_cmap())