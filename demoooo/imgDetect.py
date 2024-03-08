# 检测图片中的停车位状态,注意设置图片大小
import os
import shutil
import subprocess
import math
import pickle
import numpy as np
from PIL import Image
# 导入停车位类
from ParkingSpaceClass import ParkingSpace
# 导入必要的库和模块
import cv2
import torch
from yolov5.models.experimental import attempt_load  # YOLOv5的模型加载函数
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import letterbox
import numpy as np
from PIL import Image

def predict_image(img_path,model,device,img_width):
    # 定义输入图像尺寸
    img_size = img_width

    # 读取图片并进行缩放
    img0 = Image.open(img_path)  # 原始图片
    img0 = np.array(img0)  # 将PIL图像转换为NumPy数组

    img = letterbox(img0, new_shape=img_size)[0]

    # 转换颜色空间 RGB 到 BGR, 转换为Tensor
    img = img[:, :, ::-1].transpose(2, 0, 1)  # 注意：如果您的原始图片是RGB格式，则不需要这一步的颜色空间转换
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 前向传播
    pred = model(img, augment=False)[0]

    # 应用NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # 处理预测结果...
    print("什么狗吧")
    detections = []  # 创建一个空列表来保存检测结果
    coordinates = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = xyxy  # 解包坐标
                detections.append({
                    "coordinates": (x1.item(), y1.item(), x2.item(), y2.item()),
                    "confidence": conf.item(),
                    "class": cls.item()
                })
                data = []
                data.append(x1.item())
                data.append(y1.item())
                data.append(x2.item())
                data.append(y2.item())
                coordinates.append(data)

    return coordinates

def detect(model,device,imgPath,image_width,image_height,parking_spaces):
    # 参数设置
    #imgName = "str(order)"  # 图片名,没有路径没有.jpg
    #spacePosition = "yolov5\\spaces.p"  # 存放停车位位置的文件
    #imgPath = "images\\" + imgName + ".jpg"  # 图片路径,与parkposition.py的--img_path一致
    # image_width, image_height = 640, 640  # 图像尺寸 输入图像的宽度和高度
    # # 打开图片文件
    # image = Image.open(imgPath)
    # # 获取图片的宽度和高度
    # image_width, image_height = image.size
    # image.close()

    # 定义空列表，用于存储坐标数据
    coordinates = predict_image(imgPath,model,device,image_width)
    # 将车辆坐标存储在coordinates列表中

    # 画出停车位
    # 与parkposition.py的图片需要是同一张

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
        cv2.putText(image, str(space.id), (int(space.space_centerx), int(space.space_centery)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # 获取停车位总数
    total_parking_spaces = len(parking_spaces)
    # 统计已停车的停车位数
    occupied_spaces = sum(space.has_car for space in parking_spaces)
    # 计算剩余停车位数量
    available_spaces = total_parking_spaces - occupied_spaces

    # 在图片上显示停车位总数和剩余停车位数量
    cv2.putText(image, f"Total Spaces: {total_parking_spaces}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(image, f"Available Spaces: {available_spaces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    # 显示图片
    cv2.imshow('Parking Lot', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 在图像上绘制停车位状态，并显示结果

    # # 下面是绘制停车指示牌
    #
    # # 创建与原始图片相同大小的空白图片，背景为白色
    # blank_image = np.ones((image_height, image_width, 3), np.uint8) * 255
    #
    # # 绘制停车位状态
    # for space in parking_spaces:
    #     color = (0, 255, 0) if not space.has_car else (0, 0, 255)  # 绿色表示无车，红色表示有车
    #     vertices = np.array(space.vertices, np.int32).reshape((-1, 1, 2))
    #     cv2.fillPoly(blank_image, [vertices], color)
    #
    #     # 在空白图片上显示停车位编号
    #     cv2.putText(blank_image, str(space.id), (int(space.space_x_min), int(space.space_centery)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # # 在图片上显示停车位总数和剩余停车位数量
    # cv2.putText(blank_image, f"Total Spaces: {total_parking_spaces}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (25, 25, 55), 1)
    # cv2.putText(blank_image, f"Available Spaces: {available_spaces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (25, 25, 255), 1)
    # # 显示停车位状态指示图
    # cv2.imshow('Parking Spaces', blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return parking_spaces


if __name__ == '__main__':
    device = select_device('cpu')  # 指定运行设备
    model = attempt_load(
        'C:\\Users\\DELL\\PycharmProjects\\ParkingSpaceManagement\\demoooo\\yolov5\\runs\\train\\exp48\\weights\\best.pt',
        device=device)  # 加载模型
    model.eval()  # 设置为评估模式
    detect(model,device)


