
# DIRCN(modified)
github 주소: https://github.com/Giventicket/snu-fastmri-modified-DIRCN
<br/>
baseline model: A Densely Interconnected Network for Deep Learning Accelerated MRI(https://arxiv.org/abs/2207.02073)
<br/>
code baseline: https://github.com/JonOttesen/DIRCN
## directory

git에 공지된 최종 제출 가이드라인과 directory의 구조가 다른점 양해부탁드립니다.
```
├── DIRCN
    ├── dircn
    │   ├── base
    │   ├── config
    │   ├── dataset
    │   ├── fastmri
    │   ├── models
    │   ├── preprocessing
    │   ├── trainer
    │   ├── metrics
    │   ├── __init__.py
    │   └── logger.py
    ├── weights
    │   ├── pretrained
    │   │   ├── checkpoint.pth
    │   │   └── statistics.json
    │   ├── epoch1
    │   │   ├── checkpoint_epoch1.pth
    │   │   └── statistics.json
    │   └── epoch2 ...
    ├── train.py
    ├── eval.py
    ├── dircn.json
    ├── README.md
    └── .gitignore
        
```

## train/validation/test

## useful strategies

## modification

