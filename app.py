import os
import io
import re
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from flask import Flask
from flask import request, jsonify
from flask_cors import *
from img_process import preprocess
from predict_server import agw_predict
from predict_server import clip_test
from predict_server import video_agw_predict
from predict_server import video_ReID_attribute_information
from PIL import Image
import base64
import clip
from scipy.io import loadmat
import os.path as osp
from img_process import postprocess

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
ATTR_INFO_GF_FEATS = []
ATTR_INFO_Q_FEATS = []
ATTR_VIDEO_NAMES = []
ATTR_INFO_MODEL = None
ATTR_INFO_CAMID = []
Q_TRACKLETS = []
AITL_POOL = "avg"
AITL_SEQ_LEN = 4


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

def get_video_model(type="atrr_info"):
    attr_len = [[5, 6, 2, 2, 2, 2, 2, 2, 2], [9, 10, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    attr_loss = "bce"
    preprocess.set_seed(999)
    root = '/Users/huangtehui/PycharmProjects/re-id/data/mars'
    home_dir = 'bbox_test'
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')
    # attributes_path = osp.join(root, "mars_attributes.csv")
    # attributes = pd.read_csv(attributes_path, encoding="gbk")
    test_names = preprocess.get_names(test_name_path)
    track_test = loadmat(track_test_info_path)['track_test_info']
    query_IDX = loadmat(query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
    query_IDX -= 1  # index from 0
    track_query = track_test[query_IDX, :]
    gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
    track_gallery = track_test[gallery_IDX, :]

    pid_list = list(set(track_test[:, 2].tolist()))
    num_pids = len(pid_list)
    g_tracklets, gnum_tracklets = preprocess.process_data(track_gallery, test_names, root, home_dir)
    q_tracklets, qnum_tracklets = preprocess.process_data(track_query, test_names, root, home_dir)
    global Q_TRACKLETS
    Q_TRACKLETS = q_tracklets

    if type == "agw":
        seq_len = 6
        video_reid_model = video_agw_predict.load_video_agw_model()
        video_reid_model.eval()
    elif type == "atrr_info":
        seq_len = AITL_SEQ_LEN
        video_reid_model = video_ReID_attribute_information.load_model()
        video_reid_model.eval()
        global ATTR_INFO_MODEL
        ATTR_INFO_MODEL = video_reid_model
    print("video reid model {} loaded !".format(type))

    gf_feats = []
    g_camids = []
    g_names = []
    q_feature = None

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)

    # qf_fpath = "/Users/huangtehui/PycharmProjects/flask_apps/q_f.pkl"
    # f = open(qf_fpath, "rb")
    # qf_feats = CPU_Unpickler(f).load()
    # f.close()
    # global ATTR_INFO_Q_FEATS
    # ATTR_INFO_Q_FEATS = qf_feats

    with torch.no_grad():
        img_paths = q_tracklets[0][0]
        mp4_path = "/Users/huangtehui/PycharmProjects/flask_apps/test_images/ex.mp4"
        postprocess.image_to_video(img_paths, mp4_path)
        h265_path = "/Users/huangtehui/PycharmProjects/flask_apps/test_images/h265_ex.mp4"
        postprocess.toh256(mp4_path, h265_path)
        pid = q_tracklets[0][1]
        camid = q_tracklets[0][2]
        num = len(img_paths)
        indices_list = preprocess.get_indices(num, seq_len)
        imgs_list = []
        for indices in indices_list:
            imgs = []
            for index in indices:
                img_path = img_paths[int(index)]
                img = preprocess.read_image(img_path)
                trans_por = preprocess.get_transform()
                img = trans_por(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            imgs_list.append(imgs)
        imgs_array = torch.stack(imgs_list)
        n, s, c, h, w = imgs_array.size()
        imgs_to_device = imgs_array.to(Device)
        features, outputs = video_reid_model(imgs_to_device)
        features = features.view(n, -1)
        if AITL_POOL == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        q_feature = features.cpu()
        q_feature = q_feature.unsqueeze(0)

    # with torch.no_grad():
    #     for img_paths, pid, camid in tqdm(g_tracklets):
    #         num = len(img_paths)
    #         g_names.append((img_paths[0], img_paths[-1], num))
    #         indices_list = preprocess.get_indices(num, seq_len)
    #         imgs_list = []
    #         for indices in indices_list:
    #             imgs = []
    #             for index in indices:
    #                 img_path = img_paths[int(index)]
    #                 img = preprocess.read_image(img_path)
    #                 trans_por = preprocess.get_transform()
    #                 img = trans_por(img)
    #                 img = img.unsqueeze(0)
    #                 imgs.append(img)
    #             imgs = torch.cat(imgs, dim=0)
    #             imgs_list.append(imgs)
    #         imgs_array = torch.stack(imgs_list)
    #         n, s, c, h, w = imgs_array.size()
    #         imgs_to_device = imgs_array.to(Device)
    #         features, outputs = video_reid_model(imgs_to_device)
    #         features = features.view(n, -1)
    #         if pool == 'avg':
    #             features = torch.mean(features, 0)
    #         else:
    #             features, _ = torch.max(features, 0)
    #         features = features.cpu()
    #         gf_feats.append(features.numpy())
    #         g_camids.append(camid)
    # outputs = [torch.mean(out, 0).view(1, -1) for out in outputs]
    # preds = []
    # gts = []
    # acces = np.array([0 for _ in range(len(attr_len[0]) + len(attr_len[1]))])
    # for i in range(len(outputs)):
    #     outs = outputs[i].cpu().numpy()
    #     # outs = torch.mean(outs, 0)
    #     if attr_loss == "bce":
    #         preds.append(np.argmax(outs, 1)[0])
    #         gts.append(attrs[i].cpu().numpy()[0])
    #         acces[i] += np.sum(np.argmax(outs, 1) == attrs[i].numpy())
    # attr_metrics.update(preds, gts, acces, 1)
    # gf_feats = np.stack(gf_feats, 0)
    # g_camids = np.asarray(g_camids)

    gf_fpath = "/Users/huangtehui/PycharmProjects/flask_apps/g_f.pkl"
    f = open(gf_fpath, "rb")
    gf_feats = CPU_Unpickler(f).load()
    f.close()
    for img_paths, pid, camid in tqdm(g_tracklets):
        g_names.append((img_paths))
        g_camids.append(camid)
    g_camids = np.asarray(g_camids)
    global ATTR_INFO_GF_FEATS, ATTR_VIDEO_NAMES, ATTR_INFO_CAMID
    ATTR_INFO_GF_FEATS = gf_feats
    ATTR_VIDEO_NAMES = g_names
    ATTR_INFO_CAMID = g_camids
    m = 1
    n = len(gf_feats)
    distmat = torch.pow(q_feature, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf_feats, 2).sum(dim=1,
                                                                                                         keepdim=True).expand(
        n, m).t()
    distmat.addmm_(q_feature, gf_feats.t(), beta=1, alpha=-2)
    cosine = distmat.cpu().detach().numpy()
    id_sort = np.argsort(cosine, axis=1)
    root_p = "/Users/huangtehui/PycharmProjects/flask_apps/test_images/"
    for i, id in enumerate(id_sort[-1][:10]):
        mp4_name = str(i) + "_" + str(g_camids[id]) + "_" + str(id) + ".mp4"
        postprocess.image_to_video(list(g_names[id]), root_p + mp4_name)


@app.route('/ATTR_INFO_predict', methods=["POST"])
@cross_origin()
def ATTR_INFO_predict():
    data = request.get_json()  # 获取 JSON 数据
    pre_image_name = data["imgname"]
    in_flag = 0
    for img_paths, pid, camid in Q_TRACKLETS:
        database_root = "/Users/huangtehui/PycharmProjects/re-id/data/mars/bbox_test/" + pre_image_name[:4]
        pre_image_path = database_root + "/" + pre_image_name
        if pre_image_path in img_paths:
            in_flag = 1
            with torch.no_grad():
                num = len(img_paths)
                indices_list = preprocess.get_indices(num, AITL_SEQ_LEN)
                imgs_list = []
                for indices in indices_list:
                    imgs = []
                    for index in indices:
                        img_path = img_paths[int(index)]
                        img = preprocess.read_image(img_path)
                        trans_por = preprocess.get_transform()
                        img = trans_por(img)
                        img = img.unsqueeze(0)
                        imgs.append(img)
                    imgs = torch.cat(imgs, dim=0)
                    imgs_list.append(imgs)
                imgs_array = torch.stack(imgs_list)
                n, s, c, h, w = imgs_array.size()
                imgs_to_device = imgs_array.to(Device)
                features, outputs = ATTR_INFO_MODEL(imgs_to_device)
                features = features.view(n, -1)
                if AITL_POOL == 'avg':
                    features = torch.mean(features, 0)
                else:
                    features, _ = torch.max(features, 0)
                q_feature = features.cpu()
                q_feature = q_feature.unsqueeze(0)
                m = 1
                n = len(ATTR_INFO_GF_FEATS)
                distmat = torch.pow(q_feature, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(ATTR_INFO_GF_FEATS,
                                                                                                    2).sum(
                    dim=1,
                    keepdim=True).expand(
                    n, m).t()
                distmat.addmm_(q_feature, ATTR_INFO_GF_FEATS.t(), beta=1, alpha=-2)
                cosine = distmat.cpu().detach().numpy()
                id_sort = np.argsort(cosine, axis=1)
                root_p = "/Users/huangtehui/PycharmProjects/flask_apps/test_images/"
                h256_root_p = "/Users/huangtehui/ReactProjects/demo/public/videos/"
                temp_rs = [[], [], [], [], [], []]
                pre_rs = []
                for i, id in enumerate(id_sort[-1][:10]):
                    if temp_rs[ATTR_INFO_CAMID[id]]:
                        temp_rs[ATTR_INFO_CAMID[id]][-1].extend(list(ATTR_VIDEO_NAMES[id]))
                    else:
                        temp_rs[ATTR_INFO_CAMID[id]] = [id, ATTR_INFO_CAMID[id] + 1, list(ATTR_VIDEO_NAMES[id])]
                for i, ele in enumerate(temp_rs):
                    if ele:
                        mp4_name = str(ele[0]) + "_" + str(ele[1]) + ".mp4"
                        postprocess.image_to_video(ele[-1], root_p + mp4_name)
                        h265_mp4_name = h256_root_p + mp4_name
                        postprocess.toh256(root_p + mp4_name, h265_mp4_name)
                        pre_rs.append({"id": i + 1, "camId": str(ele[1]), "videopath": mp4_name})

                res_info = {}
                res_info["flag"] = str(in_flag)
                res_info["pre_rs"] = pre_rs
                print(res_info)
                res_body = jsonify(res_info)
                return res_body
    if not in_flag:
        res_info = {}
        res_info["flag"] = str(in_flag)
        res_info["pre_rs"] = "not in database !!!"
        print(res_info)
        res_body = jsonify(res_info)
        return res_body


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
        # print(fname)
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
    print("clip ViT-B/32 model load successfully !")


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
    # get_agw_model()
    # test_clip_intrieval()
    get_video_model()
    app.run(host=host_ip, port=host_port)
