#画出停车位(已废弃)
#与parkposition.py的图片需要是同一张
import pickle
import numpy as np
import cv2
#停车位对象
class ParkingSpace:
    def __init__(self, id, vertices, has_car, mode):
        self.id = id  #编号
        self.vertices = vertices  #顶点坐标
        self.has_car = has_car  #是否有车
        self.mode = mode  #横向还是纵向,横是0,竖是1
# 加载保存的停车位坐标文件
with open("regions.p", "rb") as f:
    total_points = pickle.load(f)
# 创建停车位对象列表
parking_spaces = []
for i, points in enumerate(total_points):
    parking_space = ParkingSpace(i, points, False, 0)
    parking_spaces.append(parking_space)
print(parking_spaces[0].vertices)

# 加载图片
image = cv2.imread('C:\\Users\\DELL\\Desktop\\parkphoto.jpg')

# 绘制停车位
for space in parking_spaces:
    vertices = np.array(space.vertices, np.int32).reshape((-1, 1, 2))
    color = (0, 0, 255) if space.has_car else (0, 255, 0)  # 绿色表示无车，红色表示有车
    cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=1)

# 显示图片
cv2.imshow('Parking Lot', image)
cv2.waitKey(0)
cv2.destroyAllWindows()