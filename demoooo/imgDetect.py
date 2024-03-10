# 检测图片中的停车位状态
import argparse
import math
import pickle

import cv2
import numpy as np
import torch

import Algorithm
from ParkingSpaceClass import ParkingSpace
from imageOperation import resize_image
from yolov5.models.experimental import attempt_load  # YOLOv5的模型加载函数
from yolov5.utils.dataloaders import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device


def predict_image(img_path, model, device, img_width):
    # 定义输入图像尺寸
    img_size = img_width

    # 读取图片并进行缩放
    img0 = resize_image(img_path, (640, 640))  # 原始图片
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
                data = [x1.item(), y1.item(), x2.item(), y2.item()]
                coordinates.append(data)

    return coordinates
#在指定起点画若干停车位,参数有:数量,起点,停车位长宽,停车位模式,方向(向下或向右),当前编号
def drawSpacesOnSpecifiedPosition(num_spaces, start_x, start_y, space_width, space_height, mode, towards, currentidx, temp_spaces, image):
    # 创建ParkingSpace对象列表
    parking_spaces = []
    if towards == 'down':
        down = 1
        right = 0
    else:
        down = 0
        right = 1
    for i in range(num_spaces):
        x1 = start_x + 10 + right * i * (space_width + 5)
        y1 = start_y + 10 + down * i * (space_height + 5)
        x2 = x1 + space_width
        y2 = y1 + space_height
        vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        hascar = temp_spaces[i+currentidx].has_car
        parking_space_fake = ParkingSpace(i + currentidx, vertices, hascar, mode)
        parking_spaces.append(parking_space_fake)
        # 在图片上绘制矩形
        if hascar:
            color = (0, 0, 255)  # 绿色表示无车，红色表示有车
        else:
            color = (0, 255, 0)
        cv2.fillPoly(image, [np.array(vertices)], color)  # 填充颜色
        cv2.polylines(image, [np.array(vertices)], isClosed=True, color=color, thickness=1)  # 画边界
        # 在停车位上标注编号
        cv2.putText(image, str(parking_space_fake.id), (int(parking_space_fake.space_x_min), int(parking_space_fake.space_centery)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return True
def detect(model, device, imgPath, image_width, image_height, parking_spaces):

    coordinates = predict_image(imgPath, model, device, image_width)
    # 将车辆坐标存储在coordinates列表中

    # 画出停车位
    # 与parkposition.py的图片需要是同一张

    parking_spaces = Algorithm.calcEverySpaceStatus(parking_spaces, coordinates)

    # 加载图片
    image = cv2.imread(imgPath)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    # 绘制停车位
    for space in parking_spaces:

        vertices = np.array(space.vertices, np.int32).reshape((-1, 1, 2))
        print(vertices)
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
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
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

    # 创建图像

    image = np.ones((image_width, image_height, 3), dtype=np.uint8) * 255
    drawSpacesOnSpecifiedPosition(9, 30, 300, 25, 15, 'horizontal', 'down', 28, parking_spaces, image)
    drawSpacesOnSpecifiedPosition(5, 90, 330, 25, 15, 'horizontal', 'down', 23, parking_spaces, image)
    drawSpacesOnSpecifiedPosition(4, 155, 350, 15, 25, 'horizontal', 'right', 19, parking_spaces, image)
    drawSpacesOnSpecifiedPosition(4, 300, 350, 15, 25, 'horizontal', 'right', 0, parking_spaces, image)
    drawSpacesOnSpecifiedPosition(5, 450, 330, 25, 15, 'horizontal', 'down', 4, parking_spaces, image)
    drawSpacesOnSpecifiedPosition(2, 480, 330, 15, 25, 'horizontal', 'down', 9, parking_spaces, image)
    drawSpacesOnSpecifiedPosition(8, 530, 300, 25, 15, 'horizontal', 'down', 11, parking_spaces, image)
    cv2.putText(image, f"Total Spaces: {total_parking_spaces}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (25, 25, 55), 1)
    cv2.putText(image, f"Available Spaces: {available_spaces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1)
    # 画箭头
    arrow_start = (image_width - 35, 70)
    arrow_end = (image_width - 35, 20)
    cv2.arrowedLine(image, arrow_start, arrow_end, (0, 0, 0), 1, tipLength=0.2)
    cv2.putText(image, f"N", (image_width - 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    # 显示图片
    cv2.imshow('Parking Spaces', image)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
    return parking_spaces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型和图片参数配置')
    parser.add_argument('--device', default='cpu', help='指定运行设备，例如 "cpu" 或 "cuda:0"')
    parser.add_argument('--model_path',
                        default='yolov5\\runs\\train\\exp48\\weights\\best.pt',
                        help='模型文件路径')
    #parser.add_argument('--img_name', default='str(order)', help='图片名，不包含路径和文件扩展名')
    parser.add_argument('--img_path', default='images\\str(order).jpg', help='图片存放的路径')
    parser.add_argument('--space_position', default='yolov5\\spaces.p', help='存放停车位位置信息的文件路径')

    args = parser.parse_args()

    device0 = select_device(args.device)  # 使用指定的设备
    model0 = attempt_load(args.model_path, device=device0)  # 加载模型
    model0.eval()  # 设置为评估模式

    #imgName = args.img_name  # 图片名
    imgPath0 = args.img_path  # 构造完整的图片路径
    spacePosition = args.space_position  # 停车位位置文件路径

    # 打开图片文件
    #image = Image.open(imgPath)
    image_for_size = cv2.imread(imgPath0)  # 注意这个图片
    image_for_size = cv2.resize(image_for_size, (640, 640), interpolation=cv2.INTER_AREA)
    # 获取图片的宽度和高度
    image_size = image_for_size.shape

    # 加载保存的停车位坐标文件
    with open(spacePosition, "rb") as f:
        total_points = pickle.load(f)
    # 打开保存停车位坐标的文件

    # 创建停车位对象列表
    parking_spaces_true = []
    for i0, points in enumerate(total_points):
        parking_space = ParkingSpace(i0, points, False, 0)

        parking_spaces_true.append(parking_space)
    # 定义停车位对象列表并初始化

    detect(model0, device0, imgPath0, image_size[0], image_size[1], parking_spaces_true)
    detect(model0, device0, imgPath0, image_size[0], image_size[1], parking_spaces_true)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
