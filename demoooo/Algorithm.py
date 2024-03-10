import math
from ParkingSpaceClass import ParkingSpace
def calcEverySpaceStatus(parking_spaces, coordinates):
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
    return parking_spaces
