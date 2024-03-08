#标记停车位:
# 默认参数:python parkposition.py  注意文件里用的图片的大小和生成文件的位置
# 自定义参数:python parkposition.py --img_path C:\\Users\\str(order).jpg --out_file yolov5/spaces.p

#打开labelimg只需要在Terminal输入labelimg
#在edge收藏夹有操作步骤https://blog.csdn.net/qq_45945548/article/details/121701492
#训练时显示PR为0,loss为NAN,pytorch版本问题去官网找合适的版本:pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#以下是步骤:
#1.先运行video2img.py(记得修改视频路径和fps)
#2.去labelimg标注图片,可以复制右边标签栏到另一张图片的标签栏,很方便(打不开的话删掉pyqt的各种包重新下载,看收藏夹里的教程),
#点击Open Dir选择C:\Users\DELL\PycharmProjects\demoooo\yolov5\VOCData\images,(根据实际改变!!!)
#点击Change Save Dir选择C:\Users\DELL\PycharmProjects\demoooo\yolov5\VOCData\Annotations(根据实际改变!!!)
#3.运行
#yolov5/VOCData/split_train_val.py (可以划分训练集和验证集和测试集比例)
#yolov5/VOCData/xml_to_yolo.py (注意将代码中的classes改成自己类别名称cars)
'''
4.在 yolov5 目录下的 data 文件夹下 新建一个 myvoc.yaml文件（可以自定义命名），用记事本打开。
内容是：
训练集以及验证集（train.txt和val.txt）的路径（可以改为相对路径）
以及 目标的类别数目和类别名称。
myvoc.yaml的内容:(根据实际修改)
train: D:/Yolov5/yolov5/VOCData/dataSet_path/train.txt
val: D:/Yolov5/yolov5/VOCData/dataSet_path/val.txt

# number of classes
nc: 1

# class names
names: ["cars"]
'''
#5.使用记事本打开 C:\Users\DELL\PycharmProjects\demoooo\yolov5\models\yolov5s.yaml。(根据实际改变!!!)把nc: xx 改成自己的类别数目
#6.终端训练代码(******先到yolov5目录下*********)
#cd ..\ParkingSpaceManagement\demoooo\yolov5\
#注意修改文件train.py和general.py实现非极大值抑制大概0.3-0.35(detect.py设置的是0.2-0.3)
#用cpu或者gpu
#python train.py --weights weights/yolov5s.pt  --cfg models/yolov5s.yaml  --data data/myvoc.yaml --epoch 50 --batch-size 20 --img 640   --device cpu
#python train.py --weights weights/yolov5s.pt  --cfg models/yolov5s.yaml  --data data/myvoc.yaml --epoch 50 --batch-size 20 --img 640   --device 0


#7.用模型检测的代码
#python detect.py --weights yolov5s.pt --source C:\Users\DELL\Desktop\parking20240103100626.mp4 --save-txt
#python detect.py --weights C:\Users\DELL\PycharmProjects\ParkingSpaceManagement\demoooo\yolov5\runs\train\exp48\weights\best.pt --source C:\Users\DELL\Desktop\parking20240103100626bk.mp4 --save-txt
#单张图片
#python detect.py --weights C:\Users\DELL\PycharmProjects\demoooo\yolov5\runs\train\exp24\weights\best.pt --source C:\Users\DELL\Desktop\094_wh860.jpg --save-txt
#8.得到结果(记得修改参数)
#运行imgdetect.py(成熟)或者videoDetect.py(未更新)
'''
在YOLOv5中，取消非极大值抑制（NMS）可以通过修改检测脚本中的参数来实现。

具体而言，在YOLOv5的detect.py脚本中，你可以找到一个名为nms的参数。将其设置为0即可取消NMS，保留所有预测框。

以下是一个示例，在调用detect.py时如何取消NMS：

shell
python detect.py --weights weights/yolov5s.pt --img 640 --conf 0.4 --nms 0
在上述示例中，--nms 0的意思是取消NMS的应用。

请注意，取消NMS可能会导致多个重叠的预测框被保留，这可能会降低检测结果的精度。因此，在实际应用中，建议根据具体需求和场景进行权衡和调整。
'''
'''
要在输出的视频文件中只加边框而不加类别名，你可以根据你的需求在 detect.py 中进行一些修改。在 YOLOv5 源代码中，输出的视频文件是在以下部分处理的：

python
if save_img or save_crop or view_img:  # Add bbox to image
    c = int(cls)  # integer class
    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    annotator.box_label(xyxy, label, color=colors(c, True))
这一部分是在处理每个检测到的物体时添加标签的地方。为了使输出的视频文件中只加边框而不加类别名，你可以将 label 设置为 None，即将标签设为空。具体做法是将上述代码中的 label 赋值部分修改为：

python
label = None
这样就可以实现在输出的视频文件里只加边框而不加类别名。请在修改完后重新运行 detect.py 文件，然后检查输出的视频文件，确认边框是否只包含边框而不包含类别名。
'''
'''
修改边框颜色为绿色(注意所有的,如果多个类别的话用原来的代码即下面的,在detect.py改的)
原代码:
if save_img or save_crop or view_img:  # Add bbox to image
    c = int(cls)  # integer class
    label = None# if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

    annotator.box_label(xyxy, label, color=colors(c, True))
修改后:
if save_img or save_crop or view_img:  # Add bbox to image
    c = int(cls)  # integer class
    label = None# if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    color = (0, 255, 0)  # 使用RGB颜色
    annotator.box_label(xyxy, label, color=color)
'''
'''
利用Softer-NMS替换原YOLOv5中的NMS，
缓解目标聚集情况下检测框处理方式不够细腻带来的漏检或误检问题，提高检测精度的同时增强算法的鲁
棒性。此外 Soft-NMS 在训练中采用传统的 NMS 方
法，仅在推断代码中实现 Soft-NMS，不增加计算量
原代码(general.py):
if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
    weights = iou * scores[None]  # box weights
    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
    if redundant:
        i = i[iou.sum(1) > 1]  # require redundancy
修改后:
if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
    if box_iou(boxes[i], boxes) > iou_thres:
        iou = math.sqrt(1-box_iou)
    else:
        iou = 1
    weights = iou * scores[None]  # box weights
    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
    if redundant:
        i = i[iou.sum(1) > 1]  # require redundancy

'''
'''
运行train.py时遇到"页面文件太小,无法完成操作"的问题,右键"此电脑","属性",右侧"高级系统设置","高级",性能的"设置","高级",虚拟内存的"更改",给D盘或E盘(pycharm所在盘)设置比较大的15G以上空间
'''