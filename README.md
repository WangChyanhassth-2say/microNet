# microNet
microNet Reappearance and Adaption  
mostly based on [liyunsheng13/micronet](https://github.com/liyunsheng13/micronet)  


## train on imagenet  
using m1 as demo
```
bash train_m1_4gpu.sh [imagenet_path] [trainlog_output_path]
```


## test on imagenet  
using m1 as demo
```
bash test_m1_4gpu.sh [imagenet_path] [testlog_output_path] [.pth_file_path]
```


## ImageNet format
```
imagenet/
│   ├── train/
│   │   └──n********/ * n
│   │       └──*.JPEG * n
│   │
│   ├── val/
│   │   └──n********/ * n
│   │       └──*.JPEG * n
```  

If the foldername is not 'train' and 'val', like 'ILSVRC2012_img_train' and 'ILSVRC2012_img_val',  
please try to rewrite the str 'train' and 'val' in the .py file  utils/dataloaders.py in line 95 and line 115  



## 2021/10/29

Update the micronet.py, using the block_a, block_b, block_c to build the model as it in the paper.


## 2021/11/02

Our research shows that using dali-cpu may not help increasing the speed,  
meanwhile using dali-gpu may run out of memory.  
So updata the utils/dataloaders.py, using pytorch_only and cut the 'if' branch in main.py.


## 2021/11/03

Adapt the structure of part of the code,  
it seems to be easier to read and adapt.


```
@article{li2021micronet,
   author    = {Li, Yunsheng and Chen, Yinpeng and Dai, Xiyang and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Liu, Zicheng and Zhang, Lei and Vasconcelos, Nuno},
   title     = {MicroNet: Improving Image Recognition with Extremely Low FLOPs},
   journal   = {arXiv preprint arXiv:2108.05894},
   year      = {2021},
}
```
