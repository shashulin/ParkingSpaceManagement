import imgDetect
import cv2
import numpy as np
from ParkingSpaceClass import ParkingSpace

# 定义图片大小
image_size = 640
# 创建ParkingSpace对象列表
parking_spaces = []
temp_spaces = imgDetect.detect()
# 车位总数
total_parking_spaces = len(temp_spaces)
#剩余车位
global available_spaces
available_spaces = 0
# 创建图像
image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255


#在指定起点画若干停车位,参数有:数量,起点,停车位长宽,停车位模式,方向(向下或向右),当前编号
def drawSpacesOnSpecifiedPosition(num_spaces, start_x, start_y, space_width, space_height, mode, towards, currentidx):

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
        parking_space = ParkingSpace(i+currentidx, vertices, hascar, mode)  # 假设所有停车位初始状态为无车
        parking_spaces.append(parking_space)

        # 在图片上绘制矩形
        if hascar:
            color = (0, 0, 255)  # 绿色表示无车，红色表示有车
        else:
            color = (0, 255, 0)
            global available_spaces
            available_spaces = available_spaces+1
        cv2.fillPoly(image, [np.array(vertices)], color)  # 填充颜色
        cv2.polylines(image, [np.array(vertices)], isClosed=True, color=color, thickness=1)  # 画边界

        # 在停车位上标注编号
        cv2.putText(image, str(parking_space.id), (int(parking_space.space_x_min), int(parking_space.space_centery)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)


    return True

drawSpacesOnSpecifiedPosition(9,30,300,25,15,'horizontal','down',28)
drawSpacesOnSpecifiedPosition(5,90,330,25,15,'horizontal','down',23)
drawSpacesOnSpecifiedPosition(4,155,350,15,25,'horizontal','right',19)
drawSpacesOnSpecifiedPosition(4,300,350,15,25,'horizontal','right',0)
drawSpacesOnSpecifiedPosition(5,450,330,25,15,'horizontal','down',4)
drawSpacesOnSpecifiedPosition(2,480,330,15,25,'horizontal','down',9)
drawSpacesOnSpecifiedPosition(8,530,300,25,15,'horizontal','down',11)
cv2.putText(image, f"Total Spaces: {total_parking_spaces}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (25, 25, 55), 1)
cv2.putText(image, f"Available Spaces: {available_spaces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)
# 显示图片
cv2.imshow('Parking Spaces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
