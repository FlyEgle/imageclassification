"""
-*- coding: utf-8 -*-
@datetime: 2021-06-28
@author  : jiangmingchao@joyy.sg
@describe: Imagenet dataset
"""
import torch
import random 
import numpy as np 
import urllib.request as urt 

from PIL import Image 
from io import BytesIO
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms as imagenet_transforms


class ImageDataset(Dataset):
    def __init__(self,
                image_file,
                train_phase,
                input_size,
                crop_size,
                shuffle = True 
                ) -> None:
        super(ImageDataset, self).__init__()
        self.image_file = image_file
        self.image_list = [x.strip() for x in open(self.image_file).readlines()]
        self.length = [x for x in range(len(self.image_list))]
        self.train_phase = train_phase
        self.input_size = input_size
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if self.shuffle and self.train_phase:
            for _ in range(10):
                random.shuffle(self.image_list)

        # train
        if self.train_phase:
            self.data_aug = imagenet_transforms.Compose([
                imagenet_transforms.RandomResizedCrop((self.crop_size, self.crop_size)),
                imagenet_transforms.RandomHorizontalFlip(),
                imagenet_transforms.ToTensor(),
                imagenet_transforms.Normalize(
                    mean= self.mean,
                    std= self.std 
                )
            ])
        # test 
        else:
            self.data_aug = imagenet_transforms.Compose([
                imagenet_transforms.Resize(self.input_size),
                imagenet_transforms.CenterCrop((self.crop_size, self.crop_size)),
                imagenet_transforms.ToTensor(),
                imagenet_transforms.Normalize(
                    mean = self.mean,
                    std = self.std 
                )
            ])

    def _decode_image(self, image_path):
        if "http" in image_path:
            image = Image.open(BytesIO(urt.urlopen(image_path).read()))
        else:
            image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image


    def __getitem__(self, index):
        for _ in range(10):
            try:
                line = self.image_list[index]
                image_path, image_label = line.split(',')[0], line.split(',')[1]
                image = self._decode_image(image_path)
                image = self.data_aug(image)
                label = torch.from_numpy(np.array(int(image_label))).long()
                return image, label 
            except Exception as e:
                index = random.choice(self.length)
                print(f"The exception is {e}, image path is {image_path}!!!")
                

    def __len__(self):
        return len(self.image_list)


# val 
class ImageDatasetTest(Dataset):
    def __init__(self,
                image_file,
                train_phase,
                input_size,
                crop_size,
                shuffle = True ,
                mode = "cnn"
                ) -> None:
        super(ImageDatasetTest, self).__init__()
        self.image_file = image_file
        self.image_list = [x.strip() for x in open(self.image_file).readlines()]
        self.length = [x for x in range(len(self.image_list))]
        self.train_phase = train_phase
        self.input_size = input_size
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.mode = mode 
        if self.shuffle and self.train_phase:
            for _ in range(10):
                random.shuffle(self.image_list)

        if self.mode == "cnn":
            self.data_aug = imagenet_transforms.Compose(
                [   
                    imagenet_transforms.Resize(self.input_size),
                    imagenet_transforms.CenterCrop(self.crop_size),
                    imagenet_transforms.ToTensor(),
                    imagenet_transforms.Normalize(
                        mean = self.mean,
                        std = self.std 
                    )
                ]
            )
        elif self.mode == "transforms":
            self.data_aug = imagenet_transforms.Compose(
                [   
                    imagenet_transforms.Resize((self.input_size, self.input_size)),
                    imagenet_transforms.ToTensor(),
                    imagenet_transforms.Normalize(
                        mean = self.mean,
                        std = self.std 
                    )
                ]
            )

    def _decode_image(self, image_path):
        if "http" in image_path:
            image = Image.open(BytesIO(urt.urlopen(image_path).read()))
        else:
            image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image


    def __getitem__(self, index):
        for _ in range(10):
            try:
                line = self.image_list[index]
                if len(line.split(',')) >= 2:
                    image_path, image_label = line.split(',')[0], line.split(',')[1]
                    label = torch.from_numpy(np.array(int(image_label))).long()
                else:
                    image_path = line
                    label = torch.from_numpy(np.array(0)).long()

                image = self._decode_image(image_path)
                image = self.data_aug(image)
                
                return image, label, image_path 
            
            except Exception as e:
                index = random.choice(self.length)
                print(f"The exception is {e}, image path is {image_path}!!!")
                

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    train_file = "/data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt"
    train_dataset = ImageDataset(
        image_file=train_file,
        train_phase=False,
        input_size = 224, 
        crop_size= 224,
        shuffle=False 
    )
    print(train_dataset)
    print(len(train_dataset))
    for idx, data in enumerate(train_dataset):
        print(f"{idx}", data[0].shape, data[1])