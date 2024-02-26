# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # 如果输出结果为 True，则表示您的计算机上的 GPU 可以被使用
        print(True)
    else:
        print(False)
    import torch

    # 如果输出结果为 True，则表示您的计算机上的 GPU 可以被使用
    print(torch.cuda.is_available())
    import torch

    print(torch.version.cuda)
    import torch

    print(torch.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
