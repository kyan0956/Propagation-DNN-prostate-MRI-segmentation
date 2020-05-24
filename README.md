# A propagation-DNN: Deep combination learning of multi-level features for MR prostate segmentation

## Usage
Please install [3D Caffe](https://au.mathworks.com/matlabcentral/answers/522143-regarding-adding-new-axis-to-the-array) first. I think you may find a greate number of tutorials talking about it.

1. Put your data to ./data folder. The data should be 64x64x32 shape alone with three dimensions. Create indexlist.txt file that annotates the file names of testing data.
2. Run demo_s1.m with Matlab to preprare necessary intermediate data for P-DNN.
3. Run demo_s2.py with Python to use P-DNN for segmentation. The results will be stored in ./result folder.

## If you think this work is helpful, please cite our paper.

@article{yan2019propagation,  
  title={A propagation-DNN: Deep combination learning of multi-level features for MR prostate segmentation},  
  author={Yan, Ke and Wang, Xiuying and Kim, Jinman and Khadra, Mohamed and Fulham, Michael and Feng, Dagan},  
  journal={Computer methods and programs in biomedicine},  
  volume={170},  
  pages={11--21},  
  year={2019},  
  publisher={Elsevier}  
}
