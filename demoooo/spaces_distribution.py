
import pickle
#导入停车位类()
from ParkingSpaceClass import ParkingSpace
spacePosition = "rc.p"#"yolov5\\spaces.p"  # 存放停车位位置的文件


# 加载保存的停车位坐标文件()
with open(spacePosition, "rb") as f:
    total_points = pickle.load(f)
# 打开保存停车位坐标的文件

# 创建停车位对象列表()
parking_spaces = []
for i, points in enumerate(total_points):
    parking_space = ParkingSpace(i, points, False, 0)
    print(parking_space.area)
    parking_spaces.append(parking_space)


# 创建一个新的图表
import matplotlib.pyplot as plt
plt.figure()

# 绘制停车位和编号
for idx, space in enumerate(parking_spaces):
    vertices = space.vertices
    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    print(x,y)
    color = 'red' if space.has_car else 'green'  # 红色表示有车，绿色表示无车
    plt.fill(x, y, color=color, alpha=0.5)  # 填充停车位区域
    plt.text(sum(x) / len(x), sum(y) / len(y), str(idx+1), color='black', fontsize=8, ha='center', va='center')  # 在停车位中心添加编号
# 获取停车位总数()
total_parking_spaces = len(parking_spaces)
# 统计已停车的停车位数()
occupied_spaces = sum(space.has_car for space in parking_spaces)
# 计算剩余停车位数量()
available_spaces = total_parking_spaces - occupied_spaces
# 添加标题和标签
plt.title('Parking Space Distribution')
plt.xlabel('X')
plt.ylabel('Y')

# 显示网格
plt.grid(False)

# 显示停车位总数和剩余停车位数量
plt.text(0, 0, f'Total Spaces: {total_parking_spaces}\nAvailable Spaces: {available_spaces}', fontsize=12)

# 显示图形
plt.show()
