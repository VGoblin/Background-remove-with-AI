import os
from skimage import io, transform
import errno
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET, U2NETP # full size version 173.6 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


def save_backremove_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGBA')

    img_name = image_name.split(os.sep)[-1]
    image = Image.open(image_name)
    image = image.convert("RGBA");
    imo = im.resize((np.array(image).shape[1],np.array(image).shape[0]),resample=Image.BILINEAR)
    filterDatas = imo.getdata()
    newFilterData = []
    for item in filterDatas:
        if(item[0] < 200 and item[1] < 200 and item[2] < 200):
            newFilterData.append((item[0], item[1], item[2], (int)(item[0] * 0.1 + item[1] * 0.8 + item[2] * 0.1)))
        else:
            newFilterData.append((item[0], item[1], item[2], 255))

    imo.putdata(newFilterData)

    pb_np = np.array(imo)
    filter = (pb_np / 255)
    output = np.int_(np.multiply(np.array(image), filter))
    # for x in range(output.shape[1]):
    #     for y in range(output.shape[0]):
    #         if(output[y][x][0] == 0 and output[y][x][1] == 0 and output[y][x][2] == 0):
    #             output[y][x][3] = 0

    output_image = Image.fromarray(np.uint8(output))
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    output_image.save(d_dir+imidx+'_backremove.png', "png")

    imo.save(d_dir+imidx+"_filter.png", "png")

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'
    # model_name = "u2net_human_seg"

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images' + '_results' + os.sep)
    # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name+'_human_seg', model_name + '_human_seg.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)

    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(torch.load(model_dir, map_location="cpu"))
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model_dir
        )

    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # print("inputs_test", type(inputs_test))
        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_backremove_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
