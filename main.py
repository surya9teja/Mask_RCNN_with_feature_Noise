import os
import torch
import torch.nn.functional as F

from typing import DefaultDict
from collections import defaultdict
from detection.utils import collate_fn

import detection.utils as utils

from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import Mask_RCNN
from torch.optim import Adam

class CocoDataset(Dataset):

    '''Implementing a Class to load COCO Dataset'''

    def __init__(self, root, annotation, transforms):

        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.img_ids = list(sorted(self.coco.imgs.keys()))[:50]
    
    def __getitem__(self, idx):
        
        img_id = self.img_ids[idx]
        anns_ids = self.coco.getAnnIds(imgIds=img_id)
        img_name = self.coco.loadImgs(ids=img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, img_name))
        anns = self.coco.loadAnns(ids=anns_ids)
        num_objs = len(anns)
        boxes = []
        labels = []
        areas = []
        masks = []

        for i in range(num_objs):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = xmin + anns[i]["bbox"][2]
            ymax = ymin + anns[i]["bbox"][3]

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]["category_id"])
            areas.append(anns[i]["area"])
            masks.append(self.coco.annToMask(anns[i]))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        is_crowd = torch.as_tensor([anns[0]["iscrowd"]], dtype=torch.int64)
        image_id = torch.tensor([img_id])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = is_crowd
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.img_ids)

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    seed(42)

    device = torch.device("cuda")
    import os
    import numpy as np
    import detection.transforms as T

    from detection.engine import train_one_epoch, evaluate
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import Subset, DataLoader
    from detection.coco_utils import get_coco

    path = "feature_noise"

    model = Mask_RCNN.maskrcnn_resnet50_fpn(pretrained=True, use_feature_noise=False)
    coco_val = get_coco('/home/surya/Mask RCNN/Mask_RCNN_with_feature_Noise/data/', "val", transforms=T.ToTensor())
    epoch = 0
    indices = torch.randperm(len(coco_val)).tolist()

    dataloaders = {}
    subsets = ["train", "val", "test"]
    subset_lens = [0.7, 0.2, 0.1]
    start_idx = 0
    for subset, subset_len in zip(subsets, subset_lens):
        idx = int(len(indices) * subset_len)
        coco_subset = Subset(coco_val, indices[start_idx:start_idx+idx])
        if subset == "train":
            batch_size = 3
            shuffle = True
        else:
            batch_size = 1
            shuffle = False
        dataloaders[subset] = DataLoader(coco_subset, batch_size, shuffle=shuffle, num_workers=0, collate_fn=utils.collate_fn)
        start_idx += idx

    model.to(device)
    optimiser = Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    writer = SummaryWriter("./Loss log/without_feature_noise")
    for epoch in range(epoch, num_epochs):
        train_one_epoch(model, optimiser, dataloaders["train"], device, epoch, writer, print_freq=1, use_feature_noise=False)
        evaluate(model, dataloaders["val"], device, epoch, use_feature_noise=False, writer=writer)