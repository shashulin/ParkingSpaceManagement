import cv2
from PIL import Image
def resize_image(image_path, output_size=(640, 640)):
    with Image.open(image_path) as img:
        resized_img = img.resize(output_size)
        return resized_img

'''
提取视频第一帧的图片
'''
def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    return image
