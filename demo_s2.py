import sys
import os
import os.path as osp
caffe_root = './caffe'
sys.path.append(caffe_root)
sys.path.append(osp.join(caffe_root, 'python'))
sys.path.append(osp.join(caffe_root, 'examples', 'pycaffe'))

os.environ['GLOG_minloglevel'] = '2'
import caffe
import tools
import surgery
import numpy as np
import setproctitle
import scipy.io as sio
import scipy.misc
from PIL import Image
import cv2
import math
import shutil
setproctitle.setproctitle(os.path.basename(os.getcwd()))

caffe.set_device(0)
caffe.set_mode_gpu()

deploy_model_file = './model/deploy.prototxt'
pretrained_file = osp.join('./model/iter_20000.caffemodel')
net = caffe.Net(deploy_model_file, caffe.TEST)
net.copy_from(pretrained_file)

batch_size = 1
indexlist = [line.rstrip('\n') for line in open('indexlist.txt')]
cnt = 0
for image_name in indexlist:
    f = open('tmp.txt', 'w')
    f.write(image_name+'\n')
    f.close()
    im = sio.loadmat(osp.join('./data/image',image_name+'.mat'))
    im = im['im_volume']
    im = im.astype(np.float32)
    im = im.transpose((2, 3, 0, 1))
    net.blobs['data'].data[0,...] = im
    cnt = cnt+1
    print 'processing '+str(cnt)+'/'+str(len(indexlist))+' data...'
    net.forward()
    prediction1 = net.blobs['prob_main'].data
    prediction4 = net.blobs['prob_sm'].data
    pp1 = prediction1[0,...][1,...]
    pp4 = prediction4[0,...][1,...]
    segMap = (pp1+pp4)/2
    sio.savemat(osp.join('./result',image_name+'.mat'),{'segMap':segMap})

shutil.rmtree('./data/image')
shutil.rmtree('./data/supervoxel')
os.remove('tmp.txt')
os.remove('currW.mat')
os.remove('sp_preference.mat')
print 'complete!'

