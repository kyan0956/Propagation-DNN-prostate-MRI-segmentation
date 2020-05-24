import caffe
import numpy as np
import scipy.io as sio
import sys
import os
import os.path as osp

class PropogationLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        params = eval(self.param_str)

        self.batch_size = params['batch_size']
        self.volume_size = params['volume_size']
        self.alpha = params['alpha']
        self.data_root = params['data_root']

    def reshape(self, bottom, top):
        top[0].reshape(
            self.batch_size, self.volume_size[2], self.volume_size[3], self.volume_size[0], self.volume_size[1])

    def forward(self, bottom, top):

        self.curr_indexlist = [line.rstrip('\n') for line in open('tmp.txt')]
        for k in range(len(self.curr_indexlist)):
            pid = self.curr_indexlist[k]
            currX = bottom[0].data[k,...]
            currX = np.squeeze(currX)
            currW = sio.loadmat(osp.join(self.data_root,'supervoxel','edge',pid+'.mat'))
            currW = currW['W']
            currSupervoxelRegionprops = sio.loadmat(osp.join(self.data_root,'supervoxel','supervoxel_regionprops',pid+'.mat'))
            currSupervoxelRegionprops = currSupervoxelRegionprops['supervoxel_regionprops']

            spnum = currW.shape[0]
            sp_preference = np.zeros((spnum),dtype=np.float32)
            for i in range(spnum):
                pixel_list = currSupervoxelRegionprops[i]['PixelList'][0]
                axis_x = pixel_list[:,0]-1
                axis_y = pixel_list[:,1]-1
                axis_z = pixel_list[:,2]-1
                tmp = currX.copy()
                tmp[axis_z,axis_x,axis_y] = 0
                sp_preference[i] = np.sum(currX-tmp)/len(axis_x)

            sio.savemat('currW',{'currW':currW})
            sio.savemat('sp_preference',{'sp_preference':sp_preference})

            dd = np.sum(currW, axis=0)
            D = np.zeros((spnum,spnum),dtype=np.float32)
            D[range(spnum),range(spnum)] = dd
            Y = np.eye(spnum,dtype=np.float32)
            Y = Y*np.spacing(1)
            D = D+Y
            aff = np.linalg.inv(np.eye(spnum,dtype=np.float32)-self.alpha*np.dot(np.linalg.inv(D),currW))
            sp_mr = np.dot(aff, sp_preference)
            curr_mr = np.ones(currX.shape, dtype=np.float32)
            for j in range(spnum):
                pixel_list = currSupervoxelRegionprops[j]['PixelList'][0]
                axis_x = pixel_list[:,0]-1
                axis_y = pixel_list[:,1]-1
                axis_z = pixel_list[:,2]-1
                curr_mr[axis_z,axis_x,axis_y] = sp_mr[j]

            top[0].data[k,...] = curr_mr[np.newaxis,...]

    def backward(self, top, propagate_down, bottom):

        self.curr_indexlist = [line.rstrip('\n') for line in open('tmp.txt')]
        for k in range(len(self.curr_indexlist)):
            pid = self.curr_indexlist[k]
            currX = bottom[0].data[k,...]
            currX = np.squeeze(currX)
            currW = sio.loadmat(osp.join(self.data_root,'supervoxel','edge',pid+'.mat'))
            currW = currW['W']
            currSupervoxelRegionprops = sio.loadmat(osp.join(self.data_root,'supervoxel','supervoxel_regionprops',pid+'.mat'))
            currSupervoxelRegionprops = currSupervoxelRegionprops['supervoxel_regionprops']

            spnum = currW.shape[0]

            dd = np.sum(currW, axis=0)
            D = np.zeros((spnum,spnum),dtype=np.float32)
            D[range(spnum),range(spnum)] = dd
            Y = np.eye(spnum,dtype=np.float32)
            Y = Y*np.spacing(1)
            D = D+Y
            daff = np.diag(np.linalg.inv(np.eye(spnum,dtype=np.float32)-self.alpha*np.dot(np.linalg.inv(D),currW)))

            curr_mr = np.ones(currX.shape, dtype=np.float32)
            for j in range(spnum):
                pixel_list = currSupervoxelRegionprops[j]['PixelList'][0]
                axis_x = pixel_list[:,0]-1
                axis_y = pixel_list[:,1]-1
                axis_z = pixel_list[:,2]-1
                curr_mr[axis_z,axis_x,axis_y] = daff[j]

            bottom[0].diff[k,...] = curr_mr[np.newaxis,...]

        bottom[0].diff[...] = bottom[0].diff*top[0].diff
