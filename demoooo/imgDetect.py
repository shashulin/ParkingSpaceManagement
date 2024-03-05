# 检测图片中的停车位状态,注意设置图片大小

import shutil
import subprocess
import math
# 导入必要的库和模块
import cv2
#参数设置
imgName = "str(order)"   # 图片名,没有路径没有.jpg
spacePosition = "yolov5\\spaces.p"  # 存放停车位位置的文件
imgPath = "images\\"+imgName+".jpg"  # 图片路径,与parkposition.py的--img_path一致
#image_width, image_height = 640, 640  # 图像尺寸 输入图像的宽度和高度
from PIL import Image
# 打开图片文件
image = Image.open(imgPath)
# 获取图片的宽度和高度
image_width, image_height = image.size
image.close()
command = "python yolov5\\detect.py --weights yolov5\\runs\\train\\exp48\\weights\\best.pt " \
          "--source " + imgPath + " --save-txt"
# 定义YOLOv5检测命令

# 执行命令获得所有车辆坐标,保存在txt文件里
subprocess.run(command, shell=True)
# 执行YOLOv5检测命令并将结果保存在文本文件中

# 定义空列表，用于存储坐标数据
coordinates = []
# 存储车辆坐标的空列表



# 打开txt文件,里面是车辆位置坐标
with open('yolov5\\runs\\detect\\exp\\labels\\'+imgName+'.txt', 'r') as file:
    # 打开YOLOv5输出的文本文件

    # 逐行读取文件内容
    for line in file:
        # 按行读取文件内容

        # 将每一行内容以空格分割，并转换为浮点数
        data = list(map(float, line.strip().split()))

        # 获取坐标信息
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
# 将车辆坐标存储在coordinates列表中

shutil.rmtree("yolov5\\runs\\detect\\exp")
# 删除YOLOv5输出的文本文件和图像文件夹

# 画出停车位
# 与parkposition.py的图片需要是同一张

import pickle
import numpy as np
import cv2

#导入停车位类
from ParkingSpaceClass import ParkingSpace



# 加载保存的停车位坐标文件
with open(spacePosition, "rb") as f:
    total_points = pickle.load(f)
# 打开保存停车位坐标的文件

# 创建停车位对象列表
parking_spaces = []
for i, points in enumerate(total_points):
    parking_space = ParkingSpace(i, points, False, 0)
    print(parking_space.area)
    parking_spaces.append(parking_space)
# 定义停车位对象列表并初始化

# 判断停车位是否有车
for space in parking_spaces:
    for idx, (x_min, y_min, x_max, y_max) in enumerate(coordinates):

        # 计算汽车中心点
        car_centerx = (x_min + x_max) / 2
        car_centery = (y_min + y_max) / 2

        # 判断停车位和汽车中心点距离
        if math.sqrt((space.space_centerx - car_centerx) ** 2 + (space.space_centery - car_centery) ** 2) < 10:
            space.has_car = True
# 判断停车位是否有车并将结果存储到停车位对象列表中

# 加载图片
image = cv2.imread(imgPath)

# 绘制停车位
for space in parking_spaces:
    vertices = np.array(space.vertices, np.int32).reshape((-1, 1, 2))
    color = (0, 0, 255) if space.has_car else (0, 255, 0)  # 绿色表示无车，红色表示有车
    cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=1)
    # 在停车位上绘制停车位编号
    cv2.putText(image, str(space.id), (int(space.space_centerx), int(space.space_centery)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# 获取停车位总数
total_parking_spaces = len(parking_spaces)
# 统计已停车的停车位数
occupied_spaces = sum(space.has_car for space in parking_spaces)
# 计算剩余停车位数量
available_spaces = total_parking_spaces - occupied_spaces

# 在图片上显示停车位总数和剩余停车位数量
cv2.putText(image, f"Total Spaces: {total_parking_spaces}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(image, f"Available Spaces: {available_spaces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# 显示图片
cv2.imshow('Parking Lot', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 在图像上绘制停车位状态，并显示结果


#下面是绘制停车指示牌

import numpy as np
import cv2

# 创建与原始图片相同大小的空白图片，背景为白色
blank_image = np.ones((image_height, image_width, 3), np.uint8) * 255

# 绘制停车位状态
for space in parking_spaces:
    color = (0, 255, 0) if not space.has_car else (0, 0, 255)  # 绿色表示无车，红色表示有车
    vertices = np.array(space.vertices, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(blank_image, [vertices], color)

    # 在空白图片上显示停车位编号
    cv2.putText(blank_image, str(space.id), (int(space.space_centerx), int(space.space_centery)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
# 在图片上显示停车位总数和剩余停车位数量
cv2.putText(blank_image, f"Total Spaces: {total_parking_spaces}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 55), 1)
cv2.putText(blank_image, f"Available Spaces: {available_spaces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 255), 1)
# 显示停车位状态指示图
cv2.imshow('Parking Spaces', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



