# DIRCN(modified)
github 주소: https://github.com/Giventicket/snu-fastmri-modified-DIRCN
<br/>
baseline model: A Densely Interconnected Network for Deep Learning Accelerated MRI(https://arxiv.org/abs/2207.02073)
<br/>
code baseline: https://github.com/JonOttesen/DIRCN
## directory

git에 공지된 최종 제출 가이드라인과 directory의 구조가 다른점 양해부탁드립니다.
최종모델은 /root/fastMRI/DIRCN/weights/best-validation/checkpoint-best.pth에 저장해뒀습니다.
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
    │   │   ├── checkpoint-epoch1.pth
    │   │   └── statistics.json
    │   ├── epoch2 ...
    │   └── best
    │   │   ├── checkpoint.pth
    │   │   └── statistics.json
    ├── train.py
    ├── recon.py
    ├── eval.py
    ├── dircn.json
    ├── README.md
    └── .gitignore
        
```

## train/validation/test
```
train data - [brain101.h5, ..., brain407.h5]
validation data - [brain1.h5, ..., brain100.h5]
test data - [brain_test1.h5, ..., brain_test58.h5]
```

## training method
```
model = DIRCN(
    num_cascades=5,
    n=16,
    sense_n=4,
    groups=4,
    sense_groups=2,
    bias=True,
    ratio=1. / 8,
    dense=True,
    variational=False,
    interconnections=True,
    )
```
1. DIRCN에 관련해서 pretrained model을 찾지못해서 위와 같이구성된 모델을 전체 주어진 set(train data, validation data)을 갖고 50 에포크를 training시켜 pretrained weight을 생성하였습니다. 이때는 단순히 train loss를 기준으로 goodness를 판단하였습니다.
2. 1.에서 얻은 pretrained model을 활용해서 train data, validation data, test data를 갖고 본격적인 training을 진행하였습니다. validation loss를 기준으로 learning rate, mini batch을 바꿔가며 train을 진행하였습니다. 이때 test data를 전부 순회하면서 reconstruction image를 생성했고 leaderboard ssim value를 평가하였습니다.
```
        freeze = []
        for k, v in self.model.state_dict().items():
            new_k = k.split(".")
            if new_k[0] == "i_cascades":
                pass
            else:
                freeze.append(k)
```
3. 위와 같이 sens_net의 weight을 freeze하고 위에서 명시된 num_cascades의 숫자를 하나씩 늘려가면서 주어진 resource(8gb gpu memory) 내에서 모델의 성능을 극대화했습니다.
4. 3.이 과적합을 일으키면 i_cascades의 weight을 freeze하고 sens_net을 훈련시켰습니다.
5. 3.과 4.를 randomly 실행하였고 num_cascades를 늘려나가는 과정에서 4.의 cuda out of memory 에러가 발생하면 3.만 진행하였습니다

## our modification
![image](https://user-images.githubusercontent.com/39179946/185732142-44dcc3fb-d541-4b9d-bbc0-222c3e613780.png)
modification과 model에 관련해서는 차후 ppt에서 더욱 자세하게 설명하도록 하겠습니다.

## how to start!(배포받은 서버기준)
```
# train(training 이후 validation과 test set에 대한 reconstruction도 차례로 진행함)
cd /root/fastMRI/DIRCN/
python train.py

# reconstruction
cd /root/fastMRI/DIRCN/
python recon.py

# evaluate
cd /root/fastMRI/DIRCN/
python eval.py

# reconstruction로 파일을 dump한 이후 evaluation을 진행하는 구조입니다.
```
