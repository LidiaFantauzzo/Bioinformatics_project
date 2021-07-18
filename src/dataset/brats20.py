import os
import nibabel as nib
import torch.utils.data as data
import numpy as np
from torch import from_numpy


class BraTSData(data.Dataset):

    def __init__(self, task='train'):

        self.task = task
        if task == 'train':
            self.image_dir = '../data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
        elif task == 'val':
            self.image_dir = '../data/Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
        else:
            self.image_dir = '../data/Brats2020/BraTS2020_TestData/MICCAI_BraTS2020_TestData'
        

        
        self.images = os.listdir(self.image_dir)
        map_classes = np.array([0,1,1,1,1])
        self.target_transform = lambda x: from_numpy(map_classes[x])

    def pad_images(self, im):
        right = np.repeat(im[:, -1], 8).reshape(240,8)
        left = np.repeat(im[:, 0], 8).reshape(240,8)
        middle = np.hstack((left, im, right))
        up = np.repeat(middle[0, :], 8).reshape(8,256)
        bottom = np.repeat(middle[-1, :], 8).reshape(8,256) 
        final = np.vstack((up, middle, bottom))
        return final
    
    def normalize(self, img):
        """Normilize image values by channels."""
        mean = np.mean(img, axis = (1,2))
        mean = np.expand_dims(mean,axis=(1,2))
        std = np.std(img, axis = (1,2))
        std = np.expand_dims(std,axis=(1,2))
        return (img - mean)/ std


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the label of segmentation.
        """
        im_suf_t1 = '_t1.nii' 
        im_suf_t1ce = '_t1ce.nii'
        im_suf_t2 = '_t2.nii'
        im_suf_flair = '_flair.nii'
        target_suf = '_seg.nii'
        im_channel = []
        
        rotate = False
        if self.task == 'train':
             p = np.random.uniform(0,1,1)
             if p>0.5:
                rotate = True

        for i in [im_suf_t1, im_suf_t1ce, im_suf_t2, im_suf_flair ]:
            image_path = os.path.join(self.image_dir, self.images[index], self.images[index]+ i) 
            img = nib.load(image_path)
            img = np.asanyarray(img.dataobj)
            if rotate:
                img  = np.rot90(img) #axis (0,1) default
            img = img[:,:,65]
            img = self.pad_images(img)
            im_channel.append(img)

        img = np.stack(im_channel,axis=0)
        target_path = os.path.join(self.image_dir, self.images[index], self.images[index]+ target_suf) 
        target = nib.load(target_path)
        target = np.asanyarray(target.dataobj)
        if rotate:
            target = np.rot90(target)
        target = target[:,:,65]
        target = self.pad_images(target)

        #### Normalize
        img = self.normalize(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.images)
