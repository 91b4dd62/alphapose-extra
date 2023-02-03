import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angleBetweenLines(l1, l2):
    x1, y1 = l1[0][0] - l1[1][0], l1[0][1] - l1[1][1]
    x2, y2 = l2[0][0] - l2[1][0], l2[0][1] - l2[1][1]
    return np.arccos((x1 * x2 + y1 * y2) / (np.sqrt(x1 * x1 + y1 * y1) * np.sqrt(x2 * x2 + y2 * y2)))

# visualization (from poseflow)


def display_pose(imgdir, visdir, track, cmap=plt.cm.get_cmap(), limit=0.03):
    print("Start visualization...\n")
    tracked = reader(track)
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

            # Determine whether the person is standing or crouching
            # Achieve this by calculating the angle between the line of hip to knee and the line of knee to ankle
            LLeg = [np.clip(pose[[11, 13], 0], 0, width),
                    np.clip(pose[[11, 13], 1], 0, height)]
            RLeg = [np.clip(pose[[12, 14], 0], 0, width),
                    np.clip(pose[[12, 14], 1], 0, height)]
            LLLeg = [np.clip(pose[[13, 15], 0], 0, width),
                     np.clip(pose[[13, 15], 1], 0, height)]
            RLLeg = [np.clip(pose[[14, 16], 0], 0, width),
                     np.clip(pose[[14, 16], 1], 0, height)]
            LAngle = angleBetweenLines(LLeg, LLLeg)
            RAngle = angleBetweenLines(RLeg, RLLeg)
            if -limit < LAngle < limit and -limit < RAngle < limit:
                raised['body'] = True
                drawLine(LLeg, 'red')
                drawLine(RLeg, 'red')
                drawLine(LLLeg, 'red')
                drawLine(RLLeg, 'red')
                drawText(np.clip(pose[[11, 13], 0], 0, width),
                         'Crouching: '+str(LAngle) + ','+str(RAngle), 'red')
            else:
                raised['body'] = False
                drawLine(LLeg, 'yellow')
                drawLine(RLeg, 'yellow')
                drawLine(LLLeg, 'yellow')
                drawLine(RLLeg, 'yellow')
                drawText(np.clip(pose[[11, 13], 0], 0, width),
                         'Not Crouching: '+str(LAngle) + ','+str(RAngle), 'yellow')

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


def drawText(pos: list, text: str, color: str):
    plt.text(pos[0], pos[1], text, color=color, fontsize=60)

