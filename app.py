import os
import io
import re
import torch
import numpy as np
from flask import Flask
from flask import request, jsonify
from flask_cors import *
from img_process import preprocess
from predict_server import agw_predict
from predict_server import clip_test
from PIL import Image
import base64
import clip

app = Flask(__name__)
CORS(app, supports_credentials=True)
AGW_MODEL = None
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GF_FEATS = []
ALL_GF_NAMES = []
ALL_CLIP_GF_IMAGES = []
CLIP_GF_FEATS = []
CLIP_VB32_MODEL = None
ENCODING = 'utf-8'


@app.route('/')
def hello_world():
    print("hellp world")
    return 'Hello World!'


# @app.before_request
# def before_first_request():
#     if not app.config['APP_ALREADY_STARTED']:
#         # 在第一个请求时执行代码
#         app.config['APP_ALREADY_STARTED'] = True
#         get_agw_model()

def get_agw_model():
    all_test_fn = []
    gf_feats = []
    global AGW_MODEL
    AGW_MODEL = agw_predict.load_model()
    AGW_MODEL.eval()
    test_image = "0003_c3s3_064744_00.jpg"
    f_img = preprocess.read_image(test_image)
    trans_por = preprocess.get_transform()
    f_ten = trans_por(f_img).unsqueeze(0)
    st_img = f_ten.to(Device)
    with torch.no_grad():
        temp_rs = AGW_MODEL(st_img)
    # qf_path = "/Users/huangtehui/PycharmProjects/re-id/toDataset/market1501/bounding_box_test"
    qf_path = "/Users/huangtehui/PycharmProjects/flask_apps/test_images/agw_test_image"
    image_fnames = [os.path.join(qf_path, x) for x in os.listdir(qf_path) if
                    ".jpg" in x or ".png" in x or ".JPG" in x or ".PNG" in x]
    pattern = re.compile(r'([-\d]+)_c(\d)')
    for f in image_fnames:
        pid, _ = map(int, pattern.search(f).groups())
        if pid == -1: continue  # junk images are just ignored
        all_test_fn.append(f)
        f_img = preprocess.read_image(f)
        f_ten = trans_por(f_img).unsqueeze(0)
        st_img = f_ten.to(Device)
        with torch.no_grad():
            temp_rs = AGW_MODEL(st_img)
        gf_feats.append(temp_rs)
        # torch.cuda.empty_cache()
    temp = torch.cat(gf_feats, dim=0)
    global GF_FEATS
    GF_FEATS = torch.nn.functional.normalize(temp, dim=1, p=2)
    global ALL_GF_NAMES
    ALL_GF_NAMES = all_test_fn
    print("agw model load successfully !")


@app.route('/AGW_predict', methods=["POST"])
@cross_origin()
def AGW_predict():
    data = request.get_json()  # 获取 JSON 数据
    imgbase64_con = data['imgbase64'].split(",")
    img = preprocess.base64_to_image(imgbase64_con[1])
    trans_por = preprocess.get_transform()
    img_ten = trans_por(img).unsqueeze(0)
    data = img_ten.to(Device)
    with torch.no_grad():
        feat = AGW_MODEL(data)
    m = 1
    n = len(GF_FEATS)
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(GF_FEATS, 2).sum(dim=1,
                                                                                                    keepdim=True).expand(
        n, m).t()
    distmat.addmm_(feat, GF_FEATS.t(), beta=1, alpha=-2)
    cosine = distmat.cpu().detach().numpy()
    id_sort = np.argsort(cosine, axis=1)
    pre_rs = []
    pattern = re.compile(r'([-\d]+)_c(\d)')
    global ENCODING
    for i, id in enumerate(id_sort[-1][:10]):
        print(ALL_GF_NAMES[id])
        pid, rs_cam_id = pattern.search(ALL_GF_NAMES[id]).groups()
        with open(ALL_GF_NAMES[id], "rb") as f:
            img_ba64 = base64.b64encode(f.read())
        f.close()
        pre_rs.append({"id": i + 1, "camId": str(rs_cam_id), "pic": img_ba64.decode(ENCODING)})
    res_info = {}
    res_info["pre_rs"] = pre_rs
    res_body = jsonify(res_info)
    return res_body


def test_clip_intrieval():
    description = "Man in blue shirt and jeans on ladder cleaning windows"
    directory_name = "/Users/huangtehui/PycharmProjects/flask_apps/test_images/clip_test_images"
    clip_model, preprocess = clip_test.load_clip_vitB32()
    clip_model.to(Device)
    original_images = []
    images = []
    for fname in os.listdir(directory_name):
        print(fname)
        if fname.endswith(".png") or fname.endswith(".jpg"):
            image = Image.open(os.path.join(directory_name, fname)).convert("RGB")
            original_images.append(image)
            images.append(preprocess(image))
    image_input = torch.tensor(np.stack(images))
    text_tokens = clip.tokenize(["This is " + description])
    image_input.to(Device)
    text_tokens.to(Device)
    clip_model.eval()
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        text_features = clip_model.encode_text(text_tokens).float()
    global CLIP_GF_FEATS
    CLIP_GF_FEATS = image_features
    global CLIP_VB32_MODEL
    CLIP_VB32_MODEL = clip_model
    global ALL_CLIP_GF_IMAGES
    ALL_CLIP_GF_IMAGES = original_images
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    print(similarity)


@app.route('/CLIP_retri', methods=["POST"])
@cross_origin()
def clip_retri():
    retri_rs = []
    data = request.get_json()  # 获取 JSON 数据
    text_des = data["description"]
    global CLIP_VB32_MODEL, CLIP_GF_FEATS
    CLIP_VB32_MODEL.eval()
    text_tokens = clip.tokenize(["This is " + text_des])
    text_tokens.to(Device)
    with torch.no_grad():
        text_features = CLIP_VB32_MODEL.encode_text(text_tokens).float()
    CLIP_GF_FEATS /= CLIP_GF_FEATS.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ CLIP_GF_FEATS.cpu().numpy().T
    id_sort = np.argsort(-similarity, axis=1)
    global ENCODING, ALL_CLIP_GF_IMAGES
    for i, id in enumerate(id_sort[-1][:5]):
        buffered = io.BytesIO()
        ALL_CLIP_GF_IMAGES[id].save(buffered, format="JPEG")
        img_ba64 = base64.b64encode(buffered.getvalue())
        retri_rs.append({"sim": str(similarity[-1][id]), "pic": img_ba64.decode(ENCODING)})
    res_info = {}
    res_info["retri_rs"] = retri_rs
    res_body = jsonify(res_info)
    return res_body


if __name__ == '__main__':
    # app.config['APP_ALREADY_STARTED'] = False  # 初始化变量
    # 后端ip
    host_ip = "127.0.0.1"
    # 端口号
    host_port = 5000
    get_agw_model()
    test_clip_intrieval()
    app.run(host=host_ip, port=host_port)
