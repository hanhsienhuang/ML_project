from opt import opt
from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import re
from pathlib import Path

class Data():
    def __init__(self):
        transform_list = [
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if opt.random_erasing:
            transform_list.append(RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]))
        train_transform = transforms.Compose(transform_list)

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = Market1501(train_transform, 'train', opt.data_path, opt.augment)
        self.testset = Market1501(test_transform, 'test', opt.data_path)
        self.queryset = Market1501(test_transform, 'query', opt.data_path)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
                                                  pin_memory=True)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path, augment=None):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        self.aug_data = []

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        def get_valid_img_path(directory):
            return [path for path in self.list_pictures(directory) if self.id(path) != -1]

        if augment is not None:
            for aug_path in augment:
                self.aug_data.extend(get_valid_img_path(aug_path))

        self.imgs = get_valid_img_path(self.data_path)
        self.imgs.extend(self.aug_data)
        print('Total number of {} image: {}'.format(dtype, len(self.imgs)))

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)
        types = ('*.jpg', '*.jpeg', '*.bmp', '*.png', '*.ppm', '*.npy')
        all_files = []
        for files in types:
            all_files.extend([str(p.absolute()) for p in Path(directory).rglob(files)])

        return all_files
