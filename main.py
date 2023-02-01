from MMEdu import MMPose
import shutil
import cv2

img = '1.jpg'


def main():
    model = MMPose(backbone='SCNet')
    result = model.inference(img=img, device='cpu', show=True)  # 在CPU上进行推理
    print(result)


if __name__ == "__main__":
    main()
