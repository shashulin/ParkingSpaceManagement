'''
将坐标文件保存的多边形在视频中画出来,并截取多边形内的部分(废弃)
'''
import cv2
import pickle
import numpy as np
#停车位对象
class ParkingSpace:
    def __init__(self, id, vertices, has_car, mode):
        self.id = id  #编号
        self.vertices = vertices  #顶点坐标
        self.has_car = has_car  #是否有车
        self.mode = mode  #横向还是纵向,横是0,竖是1
# 加载保存的停车位坐标文件
with open("parkingposition.p", "rb") as f:
    total_points = pickle.load(f)
# 创建停车位对象列表
parking_spaces = []
for i, points in enumerate(total_points):
    parking_space = ParkingSpace(i, points, False, 0)
    parking_spaces.append(parking_space)
print(parking_spaces)

# 打开视频文件
video_capture = cv2.VideoCapture("C:\\Users\\DELL\\Desktop\\parking20240103100626.mp4")

while video_capture.isOpened():
    success, frame = video_capture.read()

    if not success:
        break
    frame_count = 0
    # 在每个多边形上绘制多边形框(不画了)
    for points in total_points:
        print("yes points=")
        print(points)
        pts = np.array(points, dtype=np.int32)
        print(pts)
        print(pts[0])
        #cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # 计算最小外接矩形
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 截取最小外接矩形部分
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(frame, M, (width, height))
        cv2.imwrite(f"cropped_{frame_count}.jpg", result)
        frame_count = frame_count + 1
        cv2.imshow('Cropped Image', result)
        '''
        # 我们首先读入了原始图像，然后定义了要截取区域的四个点的坐标。通过将这些坐标转换为
        # NumPy数组，我们创建了一个与原图大小相同的空白图像。然后，我们使用
        # cv2.fillPoly()函数在空白图像上绘制多边形区域，并将指定的颜色填充到该区域内。最后，我们使用
        # cv2.bitwise_and()函数将原图和掩膜图像进行按位与操作，得到截取区域的部分。
        # 创建一个和原图等大小的空白图像
        mask = np.zeros_like(frame)

        # 使用填充颜色 (255, 255, 255) 填充指定的多边形区域
        cv2.fillPoly(mask, [points], (255, 255, 255))

        # 将原图和掩膜图像进行按位与操作，得到截取区域的部分
        cropped_image = cv2.bitwise_and(frame, mask)
        # 生成图像文件名
        filename = f"cropped_{frame_count}.jpg"
        frame_count = frame_count+1
        # 保存截取的图像
        cv2.imwrite(filename, cropped_image)
        # 显示截取后的图片
        cv2.imshow('Cropped Image', cropped_image)'''
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 显示带有多边形框的视频帧
    cv2.imshow("Video", frame)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
if __name__ == '__main__':
    print("main")