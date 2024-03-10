'''
用于标记视频帧中多边形区域的工具,并将保存的多边形区域坐标保存到pickle文件中
一个参数是图片路径,另一个是保存的坐标路径
'''
import os
import numpy as np
import cv2
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.collections import PatchCollection
from imageOperation import resize_image, extract_first_frame



points = []
prev_points = []
patches = []
total_points = []
breaker = False


class SelectFromCollection(object):
    def __init__(self, ax):
        self.canvas = ax.figure.canvas

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        global points

        points = verts
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()

        self.canvas.draw_idle()


def break_loop(event):
    global breaker
    global globSelect
    global savePath
    if event.key == 'b':
        globSelect.disconnect()
        if os.path.exists(savePath):
            os.remove(savePath)

        print("data saved in " + savePath + " file")
        with open(savePath, 'wb') as f:
            pickle.dump(total_points, f, protocol=pickle.HIGHEST_PROTOCOL)
        exit()


def onkeypress(event):
    global points, prev_points, total_points
    if event.key == 'q':
        globSelect.disconnect()
        patches.clear()

    if event.key == 'n':
        pts = np.array(points, dtype=np.int32)
        if points != prev_points and len(set(points)) == 4:
            print("Points : " + str(pts))
            patches.append(Polygon(pts))
            total_points.append(pts)
    print(points)
    prev_points = points

#标记输入视频的初始帧之一上的多边形区域。它以视频路径作为参数，并将选定多边形区域的坐标保存在pickle文件中作为输出。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="Path of image file",
                        default="images\\str(order).jpg") #与imgDetect.py文件一致
    parser.add_argument('--out_file', help="Name of the output file", default="yolov5\\spaces.p")
    parser.add_argument('--type', help="video or image", default="image")
    args = parser.parse_args()
    global globSelect
    global savePath
    savePath = args.out_file if args.out_file.endswith(".p") else args.out_file + ".p"

    print("\n> Select a region in the figure by enclosing them within a quadrilateral.")
    print("> Press the 'f' key to go full screen.")
    print("> Press the 'esc' key to discard current quadrilateral.")
    print("> Try holding the 'shift' key to move all of the vertices.")
    print("> Try holding the 'ctrl' key to move a single vertex.")
    print(
        "> After marking a quadrilateral press 'n' to save current quadrilateral and then press 'q' to start marking a new quadrilateral")
    print("> When you are done press 'b' to Exit the program\n")
    if args.type == 'image':
        image = cv2.imread(args.path)   # 注意这个图片
    elif args.type == 'video':
        image = extract_first_frame(args.path)

    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    # frames = cv2.imread('C:\\Users\\DELL\\Desktop\\R-C.jpg')  # 注意这个图片
    # new_image = frames.resize((640, 640))
    rgb_image = image[:, :, ::-1]
    while True:
        fig, ax = plt.subplots()
        image = rgb_image
        # 在每个多边形上绘制多边形框
        for points in total_points:
            pts = np.array(points, dtype=np.int32)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.polylines(image, [pts], True, (0, 255, 0), 1)

        ax.imshow(image)
        plt.get_current_fig_manager().full_screen_toggle()
        p = PatchCollection(patches, alpha=0.7)
        p.set_array(10 * np.ones(len(patches)))
        ax.add_collection(p)

        globSelect = SelectFromCollection(ax)
        bbox = plt.connect('key_press_event', onkeypress)
        break_event = plt.connect('key_press_event', break_loop)
        plt.show()
globSelect.disconnect()
