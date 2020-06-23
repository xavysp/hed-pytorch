import os
import sys
import torch
import pickle
import numpy as np
import cv2 as cv


class Logger(object):
    """ Logger class. """
    def __init__(self, path=None):
        self.console = sys.stdout
        self.file = None
        if path is not None:
            self.file = open(path, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
                self.file.close()


class AverageMeter(object):
    """ Compute and store the average and current value. """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()  # Reset the values.

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img

def save_checkpoint(state, path='./checkpoint.pth'):
    """ Save current state as checkpoint. """
    torch.save(state, path)


def load_checkpoint(net, opt, path='./checkpoint.pth', arg=None):
    """ Load previous pre-trained checkpoint.
    :param net:  Network instance.
    :param opt:  Optimizer instance.
    :param path: Path of checkpoint file.
    :return:     Checkpoint epoch number.
    """
    base_dir = arg.output
    trained_dir = os.path.join(base_dir, arg.train_dataset.lower() + '_' + arg.model_name.lower())
    path = os.path.join(trained_dir, arg.checkpoint)
    if os.path.isfile(path):
        print('=> Loading checkpoint {}...'.format(path))
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['opt'])
        return checkpoint['epoch']
    else:
        raise ValueError('=> No checkpoint found at {}.'.format(path))


def load_vgg16_caffe(net, path='./5stage-vgg.py36pickle'):
    """ Load network parameters from VGG-16 Caffe model. """
    load_pretrained_caffe(net, path, only_vgg=True)


def load_pretrained_caffe(net, path='./hed_pretrained_bsds.py36pickle', only_vgg=False):
    """ Load network parameters from pre-trained HED Caffe model. """
    # Read pretrained parameters.
    with open(path, 'rb') as f:
        pretrained_params = pickle.load(f)

    # Load parameters into network.
    print('=> Start loading parameters...')
    vgg_layers_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    for name, param in net.named_parameters():
        _, layer_name, var_name = name.split('.')
        if (only_vgg is False) or ((only_vgg is True) and (layer_name in vgg_layers_name)):
            param.data.copy_(torch.from_numpy(pretrained_params[layer_name][var_name]))
            print('=> Loaded {}.'.format(name))
    print('=> Finish loading parameters.')

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img

def restore_rgb(config,I):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: and image or a set of images
    :return: an image or a set of images restored
    """

    if  len(I)>3 and not type(I)==np.ndarray:
        I =np.array(I)
        I = I[:,:,:,0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i,...]
            x = np.array(x, dtype=np.float32)
            x += config[1]
            x = x[:, :, config[0]]
            x = image_normalization(x)
            I[i,:,:,:]=x
    elif len(I.shape)==3 and I.shape[-1]==3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    # print("The enterely I data {} restored".format(I.shape))
    return I

def cv_imshow(title='None',img=None):
    print(img.shape)
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def visualize_result(imgs_list, arg):
    """
    function for Pytorch
    :param imgs_list: a list of 8 tensors
    :param arg:
    :return:
    """
    n_imgs = len(imgs_list)
    if n_imgs==8:
        img,gt,ed1,ed2,ed3,ed4,ed5,edf=imgs_list
        img = np.transpose(np.squeeze(img),[1,2,0])
        img = restore_rgb([arg.channels_swap,arg.mean_pixel_values[:3]],img) if arg.train_dataset.lower()=='ssmihd'\
            else img
        img = np.uint8(image_normalization(img))
        h,w,c = img.shape
        gt = np.squeeze(gt)
        gt = np.uint8(image_normalization(gt))
        gt = cv.cvtColor(gt,cv.COLOR_GRAY2BGR)
        ed1 = np.squeeze(ed1)
        ed1 = np.uint8(image_normalization(ed1))
        ed1 = cv.resize(cv.cvtColor(ed1, cv.COLOR_GRAY2BGR),dsize=(w,h))
        ed2 = np.squeeze(ed2)
        ed2 = np.uint8(image_normalization(ed2))
        ed2 = cv.resize(cv.cvtColor(ed2, cv.COLOR_GRAY2BGR),dsize=(w,h))
        ed3= np.squeeze(ed3)
        ed3 = np.uint8(image_normalization(ed3))
        ed3 = cv.resize(cv.cvtColor(ed3, cv.COLOR_GRAY2BGR),dsize=(w,h))
        ed4 = np.squeeze(ed4)
        ed4 = np.uint8(image_normalization(ed4))
        ed4 = cv.resize(cv.cvtColor(ed4, cv.COLOR_GRAY2BGR),dsize=(w,h))
        ed5 = np.squeeze(ed5)
        ed5 = np.uint8(image_normalization(ed5))
        ed5 = cv.resize(cv.cvtColor(ed5, cv.COLOR_GRAY2BGR),dsize=(w,h))
        edf = np.squeeze(edf)
        edf = np.uint8(image_normalization(edf))
        edf = cv.resize(cv.cvtColor(edf, cv.COLOR_GRAY2BGR),dsize=(w,h))
        res = [img,ed1,ed2,ed3,ed4,ed5,edf,gt]
        if n_imgs%2==0:
            imgs = np.zeros((img.shape[0]*2+10,img.shape[1]*(n_imgs//2)+((n_imgs//2-1)*5),3))
        else:
            imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * ((1+n_imgs) // 2) + ((n_imgs // 2 ) * 5), 3))
            n_imgs +=1
        k=0
        imgs = np.uint8(imgs)
        i_step = img.shape[0]+10
        j_step = img.shape[1]+5
        for i in range(2):
            for j in range(n_imgs//2):
                if k<len(imgs):
                    imgs[i*i_step:i*i_step+img.shape[0],j*j_step:j*j_step+img.shape[1],:]=res[k]
                    k+=1
                else:
                    pass
        return imgs
    else:
        raise NotImplementedError
