from lib import MMPose
import sys

img = sys.argv[1]


def main():
    model = MMPose(backbone='SCNet')
    result = model.inference(img=img, device='cpu', show=True)
    print(result)


if __name__ == "__main__":
    main()
