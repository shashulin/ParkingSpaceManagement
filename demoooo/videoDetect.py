import shutil
import subprocess
import math
import pickle
import numpy as np
import cv2
import os
from PIL import Image


# 删除指定目录下的所有文件或文件夹
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


# 将视频每隔一定帧数保存为图片
def video_to_images(fps, path):
  # 打开视频文件
  cv = cv2.VideoCapture(path)
  if not cv.isOpened():
    print("\n打开视频失败！请检查视频路径是否正确\n")
    exit(0)

  # 创建存储图片的文件夹
  if not os.path.exists("images"):
    os.mkdir("images/")  # 创建文件夹
  else:
    del_file('images/')  # 清空文件夹

  order = 1  # 图片序号
  h = 0
  imgPath = "C:\\Users\\DELL\\PycharmProjects\\demoooo\\images\\str(order).jpg"
  command = "python yolov5\\detect.py --weights C:\\Users\\DELL\\PycharmProjects\\demoooo\\yolov5\\runs\\train\\exp24\\weights\\best.pt " \
            "--source " + imgPath + " --save-txt"

  # 逐帧处理视频
  while True:
    h = h + 1
    rval, frame = cv.read()

    if h == fps:
      h = 0
      order = order + 1
      if rval:
        # 调整图像大小
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        new_image = image.resize((640, 640))

        # 保存修改后的图像
        new_image.save('images/' + "str(order)" + '.jpg')
        cv2.waitKey(1)

        # 执行命令获得所有车辆坐标，保存在txt文件里
        subprocess.run(command, shell=True)

        # 定义空列表，用于存储坐标数据
        coordinates = []

        # 图像尺寸
        image_width, image_height = 640, 640

        # 打开txt文件，里面是车辆位置坐标
        with open('C:\\Users\\DELL\\PycharmProjects\\demoooo\\yolov5\\runs\\detect\\exp' + '\\labels\\str(order).txt',
                  'r') as file:
          # 逐行读取文件内容
          for line in file:
            # 将每一行内容以空格分割，并转换为浮点数
            data = list(map(float, line.strip().split()))
            temp1 = data[1]
            temp2 = data[2]
            temp3 = data[3]
            temp4 = data[4]
            # 转换为绝对坐标
            data[1] = int((temp1 - temp3 / 2) * image_width)
            data[2] = int((temp2 - temp4 / 2) * image_height)
            data[3] = int((temp1 + temp3 / 2) * image_width)
            data[4] = int((temp2 + temp4 / 2) * image_height)
            # 取出第2、3、4、5个元素，即坐标信息，并添加到列表中
            coordinates.append(data[1:5])

        # 删除临时文件夹
        shutil.rmtree("C:\\Users\\DELL\\PycharmProjects\\demoooo\\yolov5\\runs\\detect\\exp")

        # 判断停车位是否有车
        for space in parking_spaces:
          for idx, (x_min, y_min, x_max, y_max) in enumerate(coordinates):

            # 计算汽车中心点
            car_centerx = (x_min + x_max) / 2
            car_centery = (y_min + y_max) / 2

            # 判断停车位和汽车中心点距离
            if math.sqrt((space.space_centerx - car_centerx) ** 2 + (space.space_centery - car_centery) ** 2) < 10:
              space.has_car = True

        # 加载图片并绘制停车位情况
        image = cv2.imread('C:\\Users\\DELL\\PycharmProjects\\demoooo\\images\\str(order).jpg')
        for space in parking_spaces:
          vertices = np.array(space.vertices, np.int32).reshape((-1, 1, 2))
          color = (0, 0, 255) if space.has_car else (0, 255, 0)  # 绿色表示无车，红色表示有车
          cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=1)
          # 在停车位上绘制停车位编号
          cv2.putText(image, str(space.id), (int(space.space_centerx), int(space.space_centery)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示处理后的图片
        cv2.imshow('Parking Lot', image)
        cv2.waitKey(0)
      else:
        break

  cv.release()
  print('\nsave success!\n')


#导入停车位类
from ParkingSpaceClass import ParkingSpace


# 创建停车位对象列表
parking_spaces = []

# 参数设置
fps = 30  # 隔多少帧取一张图，1表示全部取
path = "C:\\Users\\DELL\\Desktop\\parking20240103100626.mp4"  # 视频路径

if __name__ == '__main__':
  # 加载保存的停车位坐标文件
  with open("yolov5\\spaces.p", "rb") as f:
    total_points = pickle.load(f)

  # 创建停车位对象并添加到列表
  for i, points in enumerate(total_points):
    parking_space = ParkingSpace(i, points, False, 0)
    parking_spaces.append(parking_space)

  # 输出第一个停车位的顶点坐标
  print(parking_spaces[0].vertices)

  # 调用函数处理视频并保存图片
  video_to_images(fps, path)

# 会在代码的当前文件夹下 生成images文件夹 用于保存图片
  # 如果有images文件夹，会清空文件夹！
