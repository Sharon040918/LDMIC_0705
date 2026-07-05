import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from pathlib import Path
import numpy as np
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as TF


try:
    from parahydra_plus.lib.mydatatransform import CutMix
except ImportError:
    class CutMix:
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def __call__(self, images):
            return images


try:
    from parahydra_plus.lib.utils import MinimalCrop
except ImportError:
    class MinimalCrop:
        def __init__(self, min_div=64):
            self.min_div = min_div

        def __call__(self, image):
            _, h, w = image.shape
            h_new = h - (h % self.min_div)
            w_new = w - (w % self.min_div)
            if h_new == h and w_new == w:
                return image

            top = (h - h_new) // 2
            left = (w - w_new) // 2
            return image[:, top:top + h_new, left:left + w_new]



class MultiCameraImageDataset(Dataset):
    def __init__(self, ds_type='train', ds_name='wildtrack', root='D:/LDMIC/LDMIC-main/data', crop_size=(256, 256), num_camera=3, dir_num = 2,
                 train_test_ratio = 4, dataaug = False ,dataaug_alpha = 1.0, force_crop = False, **kwargs):
        super().__init__()
        self.crop_size = crop_size
        self.force_crop = force_crop
        self.path = Path(f"{root}") / ds_name
        self.ds_name = ds_name
        self.ds_type = ds_type
        self.train_test_ratio = 1.0 * train_test_ratio / (train_test_ratio + 1)

        
        self.if_dataaug = dataaug
        self.data_aug = CutMix(dataaug_alpha)
        self.num_camera = num_camera
        self.dir_num = dir_num
        self.image_lists = self.get_files() #* self.image_lists 是一个长度为 num_camera 的列表，每个元素是一个子列表，包含该摄像头拍摄的图像路径  
            
        if ds_type == "test":
            self.crop = MinimalCrop(min_div=64)
        else:
            self.crop = None
        if self.ds_type=="train":    #* ds_type == "train" 时，使用数据增强技术（随机裁剪、水平翻转、垂直翻转等）。
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(self.crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        elif self.force_crop: 
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.RandomCrop(self.crop_size)])
        else:   #* ds_type == "test" 时，只进行张量转换.
            self.transform = transforms.Compose([transforms.ToTensor()])
        print(f'Loaded {ds_type} dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

    def __len__(self):
        return len(self.image_lists[0])

    def __getitem__(self, index):
        """_summary_
        在每次迭代时，DataLoader 会根据 batch_size 和 shuffle 参数生成一组索引。

        对于每个索引，DataLoader 会调用 dataset.__getitem__(index) 获取对应的样本。
        """

        # 用 torchvision 更快读取：返回 uint8, CHW, RGB
        imgs = [read_image(self.image_lists[i][index], mode=ImageReadMode.RGB) for i in range(self.num_camera)]

        # 你原来的 test 裁剪逻辑保持：MinimalCrop 还是吃 PIL，就临时转一下
        if self.crop is not None:
            imgs = [self.crop(x) for x in imgs]   # 直接 tensor 裁剪

        # 转成 numpy(H,W,C) 以便保持你后面的 np.concatenate + transforms.ToTensor 流程不变
        images = np.concatenate([x.permute(1, 2, 0).contiguous().numpy() for x in imgs], axis=-1)
        
        #* image_list 是 一个长度为self.num_camera的列表，列表中的每一个元素的大小为(H_crop, W_crop, 3)
        images = torch.chunk(self.transform(images), self.num_camera)
        #* image_list 是 一个长度为self.num_camera的列表，列表中的每一个元素的大小为(3， H_crop, W_crop)
        if(self.if_dataaug and self.ds_type == 'train'):
            images = self.data_aug(images)
        return images


    def get_files(self):
        dataset_lists = [[] for i in range(self.num_camera)]   #* 这行代码创建了一个包含 num_camera 个空列表的列表，num_camera 默认为 7。每个空列表用于存储一个摄像头的图像路径。
        image_lists = [[] for i in range(self.dir_num)]
        train_lists = []
        test_lists = []
        
        for idx in range(1, self.dir_num + 1):
            path = self.path / f"C{idx}"
            for image_path in path.iterdir():  # 更稳妥的做法
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # 遍历当前路径下所有符合扩展名的图片
                    image_lists[idx - 1].append(str(image_path))    # 将图片路径添加到对应摄像头的列表中

                    
        for idx in range(1, self.dir_num + 1):
            train_num = (int) (self.train_test_ratio * len(image_lists[idx-1]))
            #* actual train sample number is : train_num - num_camera + 1
            #* 
            if((train_num - self.num_camera + 1) % 2 == 1):
                train_num = train_num + 1

            train_lists.append(image_lists[idx-1][0:train_num])
            test_lists.append(image_lists[idx-1][train_num:])

        if self.ds_type == 'train':
            for idx in range(1, self.dir_num + 1):
                cun_len = len(train_lists[idx-1])
                for ii in range(cun_len-self.num_camera+1):
                    for jj in range(self.num_camera):
                        dataset_lists[jj].append(train_lists[idx-1][ii+jj])
        else:
            for idx in range(1, self.dir_num + 1):
                cun_len = len(test_lists[idx-1])
                for ii in range(cun_len-self.num_camera+1):
                    for jj in range(self.num_camera):
                        dataset_lists[jj].append(test_lists[idx-1][ii+jj])
        return dataset_lists
    

class StereoImageDataset(Dataset):
    """_summary_
    AdaptiveMultiCameraImageDataset 是一个自定义的 PyTorch Dataset 类，用于加载和处理多摄像头（multi-camera）图像数据集，支持训练和测试模式，
    并可适应不同数量的摄像头图像。该类可以用于训练深度学习模型，特别是与多视角图像相关的任务。
    """
    def __init__(self, ds_type='train',ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256),train_test_ratio = 4,
                 dataaug = True, dataaug_alpha= 1.0, force_crop = False,
                 **kwargs):
        """_summary_
        功能概述：
        该函数初始化了数据集对象，并根据不同的参数和数据集类型（训练或测试）设置相应的处理流程。
        
        参数：

        参数	说明
        ds_type	数据集类型，可选值为 'train' 或 'test'，决定数据加载方式。
        ds_name	数据集名称（默认为 wildtrack）。
        root	数据集根路径，指向数据存储目录。
        crop_size	随机裁剪大小（默认为 (256, 256)）。
        **kwargs	允许传递额外的参数。
        """
        super().__init__()
        self.crop_size = crop_size
        self.force_crop = force_crop
        self.path = Path(f"{root}") / ds_name#* 指定数据集的根目录。
        self.ds_type = ds_type  #* self.ds_type 表示 是“训练数据集” 还是 测试数据集

        #* 定义了self.transform， 用于对图像进行变换
        self.num_camera = 2   #*  调用 self.set_num_camera() 随机设置可用的摄像头数量。
        
        self.if_dataaug = dataaug
        self.data_aug = CutMix(dataaug_alpha)
        self.train_test_ratio = 1.0 * train_test_ratio / (train_test_ratio + 1)
        self.image_lists = self.get_files() #* self.image_lists 是一个  长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。
        #* self.image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
        #* 使用 self.get_files() 获取数据文件路径，并根据数据集类型设置不同的处理方式。
        if ds_type == "test":   
            #*  如果是测试集 (ds_type == "test")，使用 MinimalCrop 对图像进行裁剪。
            self.crop = MinimalCrop(min_div=64)
        else:
            self.crop = None
        if self.ds_type=="train":    #* ds_type == "train" 时，使用数据增强技术（随机裁剪、水平翻转、垂直翻转等）。
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(self.crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        elif self.force_crop: 
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.RandomCrop(self.crop_size)])
        else:   #* ds_type == "test" 时，只进行张量转换.
            self.transform = transforms.Compose([transforms.ToTensor()])

        print(f'Loaded dataset from {self.path}. Found {len(self.image_lists[0])} files.')
        print(f'Using {len(self.image_lists)} cameras.')
    def __len__(self):
        """_summary_
        功能： 
        返回数据集中的样本数量，即图像文件的数量。
        这里返回 self.image_lists[0] 的长度，因为它包含了第一个摄像头（C1）的图像路径，其他摄像头的路径数量应该相同。
        """
        return len(self.image_lists[0])

    def __getitem__(self, index):
        """_summary_
        整体介绍
        __getitem__ 函数是 Dataset 类的核心方法，用于从数据集中获取一个样本。
        其主要功能是根据数据集类型（训练或测试），加载指定数量的摄像头图像并进行一系列必要的图像处理（如裁剪、转换等）。
        最终，它返回一个包含处理后的图像的张量列表。

        分析函数参数
        index:
        该参数是从数据集中获取图像样本时的索引值。
        在 __getitem__ 中，index 用来访问 self.image_lists 中对应位置的图像路径。index 是一个整数值，表示要从 image_lists 中选取的图像的索引。
        
        
        返回值:

        frames: 这是一个包含多个图像张量的tuple，每个元素代表一个摄像头拍摄的图像。
        图像已经过必要的预处理操作（如裁剪、转换等）。每个图像被切割成多个部分（根据 num_camera），并将它们组合成一个tuple。
        #* frames 维度为: (self.num_camera,C, H,W); frames[i] 表示 第i个视点的第index个图像所对应的张量
        
        """
            #* 如果是训练集，使用 self.images_index（随机选取的摄像头索引）加载对应的图像
            #* self.images_index 是一个 长度为 self.num_camera 的列表， self.images_index[i] 的取值为[0~7]的一个随机数，且互相不重复
            #* self.image_lists[i] 表示 摄像头i拍摄的所有图像的路径，是一个列表； self.image_lists[i][index] 就是摄像头i拍摄的第index个图像
            #* 每个图像都使用 PIL.Image.open 打开，并通过 .convert('RGB') 转换为 RGB 格式。
            #* 如果是测试集，加载所有 num_camera 个摄像头的图像。

        imgs = [read_image(self.image_lists[i][index], mode=ImageReadMode.RGB) for i in range(self.num_camera)]

        # 你原来的 test 裁剪逻辑保持：MinimalCrop 还是吃 PIL，就临时转一下
        if self.crop is not None:
            imgs = [self.crop(x) for x in imgs]   # 直接 tensor 裁剪
        images = np.concatenate([x.permute(1, 2, 0).contiguous().numpy() for x in imgs], axis=-1)   #* np.concatenate(..., axis=-1) 将所有图像沿最后一个维度（通道维度）拼接。

        #* image_list 是 一个长度为self.num_camera的列表，列表中的每一个元素的大小为(H_crop, W_crop, 3)
        images = torch.chunk(self.transform(images), self.num_camera)
        #* image_list 是 一个长度为self.num_camera的列表，列表中的每一个元素的大小为(3， H_crop, W_crop)
        if(self.if_dataaug and self.ds_type == 'train'):
            images = self.data_aug(images)
        return images

    def get_files(self):
        """_summary_
        整体介绍
        get_files 函数用于根据数据集名称和数据集类型（训练或测试），从指定的路径加载图像文件路径。它从 WildTrack 数据集中的多个摄像头视角获取图像路径，并按摄像头的不同存储图像路径到一个列表中，最后返回这些路径列表。

        分析函数参数
        num_camera (默认值 7):

        该参数指定数据集中使用的摄像头数量。通常为 7，代表 WildTrack 数据集中有 7 个不同的摄像头视角。它用于初始化一个包含多个摄像头图像路径的列表 image_lists，每个摄像头的路径都将存储在列表中的不同子列表中。
        分析函数返回值
        
        返回值:

        image_lists: 该函数返回一个长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
        每个摄像头（即 C1, C2, ..., C7）都会有自己对应的图像路径列表。假设有 7 个摄像头，image_lists 的长度为 7，每个子列表分别包含该摄像头的图像路径。
        """
        image_lists = [[] for i in range(self.num_camera)]   #* 这行代码创建了一个包含 num_camera 个空列表的列表，num_camera 默认为 7。每个空列表用于存储一个摄像头的图像路径。
        # for idx in range(1, self.num_camera + 1):
        #     path = self.path / f"C{idx}" 
        #     for ext in ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]:  # 添加大写扩展名
        #         for image_path in path.glob(ext):  # 遍历当前路径下所有符合扩展名的图片
        #             image_lists[idx - 1].append(str(image_path))  # 将图片路径添加到对应摄像头的列表中
                    
        for idx in range(1, self.num_camera + 1):
            path = self.path / f"C{idx}"
            for image_path in path.iterdir():  # 更稳妥的做法
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # 遍历当前路径下所有符合扩展名的图片
                    image_lists[idx - 1].append(str(image_path))    # 将图片路径添加到对应摄像头的列表中
        for idx in range(self.num_camera):
            train_num = (int) (self.train_test_ratio * len(image_lists[idx]))
            if(train_num % 2 == 1):
                train_num = train_num + 1            
            if(self.ds_type == 'train'):
               image_lists[idx] = image_lists[idx][0:train_num]
            else:
                image_lists[idx] = image_lists[idx][train_num:]

        return image_lists #* image_lists 将被返回，它是一个长度为 num_camera（默认为 7）的列表，每个元素是一个子列表，包含该摄像头拍摄的图像路径

