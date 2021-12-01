# microNet
microNet Recurrence and Adaption  
mostly based on [liyunsheng13/micronet](https://github.com/liyunsheng13/micronet)  


## test the performance of the model
you may code like this  
```
import torch
from micronet import *
from torchsummaryX import summary

model = micronet('m1')
dummy_input = torch.rand(32,3,224,224)
profile = summary(model, dummy_input)
```
or  
```
python3 micronet.py
```

## train on imagenet  
using m1 as demo
```
python3 -W ignore main.py --arch m1 -d ./imagenet -c ./output/m1
```


## test on imagenet  
using m1 as demo
```
python3 -W ignore main.py --arch m1 -d ./imagenet -c ./output/m1_test -e --weight ./output/m1/model_best.pth
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

## 2021/11/04

Pull the micronet out and change some default args.  
Now it can be tested singly,   
The performance_test code was shown above.

## 2021/11/05

Update the DyShiftMax activation,  
add lambdas and alphas on it.  
the acc might rises but the mflops rise either.  
for m1 as example, the Mult-Adds rises from 5.244M to 5.688M.

### Ready to put the model into training :)

## 2021/11/06

Fix some bugs encountered when we train on distributed parallel with the updated version.  

## 2021/11/07

### Results:
training on imagenet_t:  

| Model | Param | MAdds | Acc@1 | Acc@5 |
| ----- | ----- | ----- | ----- | ----- |
| microNet_m1 | 1.700M | 5.244M | 55.385 | 78.714 |
| microNet_m1_updated | 1.700M | 5.688M | 57.648 | 80.330 |
| ghostNet_width=0.1 | 1.442M | 6.911M | 51.407 | 78.538 |

## 2021/11/30
the loss of rvm_micronet is not good as expect,  
so I try to rewrite the backbone of micronet.  
add ghost_module, SE_module and so on...  
the mflops even lighter than usual...  
put it on the coeus later, hope it performs well.  

# Citation
```
@article{li2021micronet,
   author    = {Li, Yunsheng and Chen, Yinpeng and Dai, Xiyang and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Liu, Zicheng and Zhang, Lei and Vasconcelos, Nuno},
   title     = {MicroNet: Improving Image Recognition with Extremely Low FLOPs},
   journal   = {arXiv preprint arXiv:2108.05894},
   year      = {2021},
}
```
