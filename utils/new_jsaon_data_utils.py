import json
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np
import os

def crop_center(data, out_sp):
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[0] - out_sp[0]) / 2)
    y_crop = int((in_sp[1] - out_sp[1]) / 2)
    z_crop = int((in_sp[2] - out_sp[2]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

class AgeData(Dataset):
    def __init__(self, json_file, root_dir, split='training', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(json_file, 'r') as f:
            data_dict = json.load(f)
        self.datalist = data_dict[split]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        item = self.datalist[index]
        img_path = os.path.join(self.root_dir, item["image"])
        img = sitk.ReadImage(img_path)
        img2 = sitk.GetArrayFromImage(img)
        img3 = crop_center(img2, (96,112,96))
        label = item["label"]
        sample = {'image': img3, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample