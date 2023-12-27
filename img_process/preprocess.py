import io
import os.path as osp
import base64
from PIL import Image
import torchvision.transforms as T


def base64_to_image(base64_code):
    """将base64的数据转换成rgb格式的图像矩阵"""
    img_data = base64.b64decode(base64_code)
    # img_array = np.frombuffer(img_data, np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = Image.open(io.BytesIO(img_data))
    return img


def get_transform():
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        normalize_transform
    ])
    return eval_transforms

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img