# CoSiamRPN
## Experiments
### OTB Benchmark
#### Comparison to SOTA 
- Model Size vs AUC on OTB2013  

| Trackers | CoSiamRPN(E) | CoSiamRPN(R) | SiamRPN++(R) | SiamRPN++(A) | SiamFC++(G) | DaSiamRPN(A) | SiamFC(R) |
| :------: | :----------: | :----------: | :----------: | :----------: | :---------: | :----------: | :-------: |
|  Params  |     8.4      |     5.9      |     54.0     |     6.3      |    15.5     |     22.6     |    2.3    |
|  FLOPs   |     5.1      |     5.5      |     59.7     |     5.6      |    16.6     |     5.8      |    4.3    |
|   AUC    |    0.714     |    0.696     |    0.691     |    0.666     |    0.695    |    0.656     |   0.672   |

- OTB2015
  _content_ 

#### Ablation analysis  on OTB100

|    Backbone     |     RPN      |    CoRPN     |     MAN      | AUC$\uparrow$ |
| :-------------: | :----------: | :----------: | :----------: | :-----------: |
|   ResNet50-C3   | $\checkmark$ |              |              |     0.669     |
|   ResNet50-C3   |              | $\checkmark$ |              |     0.674     |
|   ResNet50-C3   | $\checkmark$ |              | $\checkmark$ |     0.673     |
|   ResNet50-C3   |              | $\checkmark$ | $\checkmark$ |     0.679     |
| EfficientNet-B0 | $\checkmark$ |              | $\checkmark$ |     0.684     |
| EfficientNet-B0 |              | $\checkmark$ | $\checkmark$ |     0.692     |

https://github.com/votchallenge/trax
### VOT Benchmark
#### VOT2018
| Trackers  | A($\uparrow$) | R($\downarrow$) | EAO($\uparrow$) |
| :-------: | :-----------: | :-------------: | :-------------: |
|   LADCF   |     0.503     |      0.159      |      0.389      |
|    MFT    |     0.505     |      0.140      |      0.385      |
|  SiamRPN  |     0.586     |      0.276      |      0.383      |
|   UPDT    |     0.536     |      0.184      |      0.378      |
|    RCO    |     0.507     |      0.155      |      0.376      |
|    DRT    |     0.519     |      0.201      |      0.356      |
| DeepSTRCF |     0.523     |      0.215      |      0.345      |
|    CPT    |     0.339     |      0.239      |      0.339      |
| SA_Siam_R |     0.337     |      0.258      |      0.337      |
|  DLST++   |     0.325     |      0.224      |      0.325      |
| SiamRPN++ |     0.600     |      0.234      |      0.414      |
|   ATOM    |     0.590     |      0.204      |      0.401      |
| CoSiamRPN |     0.601     |      0.173      |      0.426      |

#### VOT2019
| Trackers | CoSiamRPN | DiMP  | SiamRPN-DW | ATOM  | SiamMask | SiamRPN++ |  SPM  | SA_SIAM | SiamRPNX | Siamfcos | iourpn |
| :------: | :-------: | :---: | :--------: | :---: | :------: | :-------: | :---: | :-----: | :------: | :------: | :----: |
|   EAO    |   0.314   | 0.379 |   0.299    | 0.292 |  0.287   |   0.285   | 0.275 |  0.253  |  0.224   |  0.223   | 0.161  |
|  Speed   |   71.2    |  4.2  |    12.8    |  4.1  |   71.0   |   37.1    | 33.2  |  25.3   |   22.3   |   1.4    |  20.3  |


### LaSOT



