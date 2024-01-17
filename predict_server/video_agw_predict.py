import torch
import models


def load_video_agw_model():
    arch = 'AGW_Plus_Baseline'
    class_num = 625
    print("Initializing model: {}".format(arch))
    model = models.init_model(name=arch, num_classes=class_num, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    load_model = "/Users/huangtehui/PycharmProjects/re-id/mars_agw_plus/checkpoint_ep400.pth.tar"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = torch.load(load_model, map_location=torch.device(device))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    start_epoch = pretrained_model['epoch'] + 1
    best_rank1 = pretrained_model['rank1']
    model.to(device)
    return model
