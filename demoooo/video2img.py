#将视频每隔30帧存一张图片到某个yolov5/VOCData/images文件夹中

import cv2
import os
from PIL import Image
#此删除文件夹内容的函数来源于网上
def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

def video_to_images(fps,path):
  cv = cv2.VideoCapture(path)
  if(not cv.isOpened()):
    print("\n打开视频失败！请检查视频路径是否正确\n")
    exit(0)
  if not os.path.exists("yolov5/VOCData/images"):
    os.mkdir("yolov5/VOCData/images/") # 创建文件夹
  else:
    del_file('yolov5/VOCData/images/') # 清空文件夹
  order = 0   #序号
  h = 0
  while True:
    h=h+1
    rval, frame = cv.read()
    if h == fps:
      h = 0
      order = order + 1
      if rval:
        # 调整图像大小
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        new_image = image.resize((640, 640))

        # 保存修改后的图像
        new_image.save('yolov5/VOCData/images/' + str(order) + '.jpg')
        cv2.waitKey(1)
      else:
        break
  cv.release()
  print('\nsave success!\n')

# 参数设置
fps = 30   # 隔多少帧取一张图  1表示全部取
path="C:\\Users\\DELL\\Desktop\\parking20240103100626.mp4" # 视频路径 比如 D:\\images\\tram_result.mp4 或者 D:/images/tram_result.mp4

if __name__ == '__main__':
  video_to_images(fps,path)
  # 会在代码的当前文件夹下 生成images文件夹 用于保存图片
  # 如果有images文件夹，会清空文件夹！
