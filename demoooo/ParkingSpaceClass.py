#停车位类
class ParkingSpace:
    def __init__(self, id, vertices, has_car, mode):
        self.id = id  # 停车位编号
        self.vertices = vertices  # 停车位四点坐标
        self.has_car = has_car  # 停车位是否有车
        self.mode = mode  # 当前模式(开放?横竖?)

        # 计算停车位面积
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        x4, y4 = vertices[3]

        # 计算向量AB和向量AC的叉积
        S1 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        # 计算向量AD和向量AC的叉积
        S2 = (x4 - x1) * (y3 - y1) - (x3 - x1) * (y4 - y1)

        # 计算四边形面积
        area = abs(S1 + S2) / 2

        # 将面积存入对象属性中
        self.area = area

        # 计算停车位中心点
        space_x_min = min(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
        space_y_min = min(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
        space_x_max = max(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
        space_y_max = max(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
        self.space_centerx = (space_x_min + space_x_max) / 2
        self.space_centery = (space_y_min + space_y_max) / 2

