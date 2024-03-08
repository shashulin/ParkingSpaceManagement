from PIL import  Image
def resize_image(image_path, output_size=(640, 640)):
    with Image.open(image_path) as img:
        resized_img = img.resize(output_size)
        return resized_img
