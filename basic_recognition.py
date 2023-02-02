import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def angle(v1, v2):
    return np.abs(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


def angleBetweenLines(line1, line2):
    # line1 = [[x1, y1], [x2, y2]]
    # line2 = [[x3, y3], [x4, y4]]
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    v1 = [x2 - x1, y2 - y1]
    v2 = [x4 - x3, y4 - y3]
    return angle(v1, v2)

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

            #                    0       1       2       3        4        5             6          7
            coco_part_names = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow',
                               'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
            #                     8          9         10      11      12       13       14       15        16
            colors = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y',
                      'y', 'y', 'y', 'g', 'g', 'g', 'g', 'g', 'g']
            pairs = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [
                8, 10], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [6, 12], [5, 11]]

            raised = {}

            LBody = [np.clip(pose[[5, 11], 0], 0, width),
                     np.clip(pose[[5, 11], 1], 0, height)]
            RBody = [np.clip(pose[[6, 12], 0], 0, width),
                     np.clip(pose[[6, 12], 1], 0, height)]

            # Calculate the average line of the line LBody and the line RBody
            Body = [[(LBody[0][0] + RBody[0][0]) / 2, (LBody[0][1] + RBody[0][1]) / 2],
                    [(LBody[1][0] + RBody[1][0]) / 2, (LBody[1][1] + RBody[1][1]) / 2]]
            drawLine(Body, 'green')
            stdAngle = angle(Body[0], Body[1])

            # Determine whether the person's left upper arm is raised
            # If the angle between the left ankle to shoulder and the standard angle is greater than 90 degrees, it is considered to be raised
            LArm = [np.clip(pose[[5, 7], 0], 0, width),
                    np.clip(pose[[5, 7], 1], 0, height)]
            if angleBetweenLines(LArm, Body) > 1.57:
                drawLine(LArm, 'yellow')
                raised['LArm'] = True
            else:
                drawLine(LArm, 'green')
                raised['LArm'] = False

            # Same for the right upper arm
            RArm = [np.clip(pose[[6, 8], 0], 0, width),
                    np.clip(pose[[6, 8], 1], 0, height)]
            if angleBetweenLines(RArm, Body) > 1.57:
                drawLine(RArm, 'yellow')
                raised['RArm'] = True
            else:
                drawLine(RArm, 'green')
                raised['RArm'] = False

            # Determine whether the person's left lower arm is raised
            # If the angle between the left wrist to elbow and the upper arm is lower than 90 degrees, it is considered to be raised, unless the upper arm is raised
            LLArm = [np.clip(pose[[7, 9], 0], 0, width),
                     np.clip(pose[[7, 9], 1], 0, height)]
            if raised['LArm']:
                if angleBetweenLines(LLArm, LArm) > 1.57:
                    drawLine(LLArm, 'yellow')
                    raised['LLArm'] = True
                else:
                    drawLine(LLArm, 'green')
                    raised['LLArm'] = False
            else:
                if angleBetweenLines(LLArm, LArm) > 1.57:
                    drawLine(LLArm, 'green')
                    raised['LLArm'] = False
                else:
                    drawLine(LLArm, 'yellow')
                    raised['LLArm'] = True

            # Same for the right lower arm
            RLArm = [np.clip(pose[[8, 10], 0], 0, width),
                     np.clip(pose[[8, 10], 1], 0, height)]
            if raised['RArm']:
                if angleBetweenLines(RLArm, RArm) > 1.57:
                    drawLine(RLArm, 'yellow')
                    raised['RLArm'] = True
                else:
                    drawLine(RLArm, 'green')
                    raised['RLArm'] = False
            else:
                if angleBetweenLines(RLArm, RArm) > 1.57:
                    drawLine(RLArm, 'green')
                    raised['RLArm'] = False
                else:
                    drawLine(RLArm, 'yellow')
                    raised['RLArm'] = True

            # Draw the keypoints
            for idx_c, color in enumerate(colors):
                plt.plot(np.clip(pose[idx_c, 0], 0, width), np.clip(pose[idx_c, 1], 0, height), marker='o',
                         color=color, ms=80/alpha_ratio*np.mean(pose[idx_c, 2]), markerfacecolor=(1, 1, 0, 0.7/alpha_ratio*pose[idx_c, 2]))
            for idx in range(len(pairs)):
                plt.plot(np.clip(pose[pairs[idx], 0], 0, width), np.clip(pose[pairs[idx], 1], 0, height), 'r-',
                         color=cmap(tracked_id), linewidth=60/alpha_ratio*np.mean(pose[pairs[idx], 2]), alpha=0.1/alpha_ratio*np.mean(pose[pairs[idx], 2]))
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


def drawLine(line: list, color: str):
    plt.plot(line[0], line[1], 'r-', linewidth=20, color=color)


display_pose('examples/', 'examples/vis/',
             reader('examples/output.json'), plt.cm.get_cmap())
