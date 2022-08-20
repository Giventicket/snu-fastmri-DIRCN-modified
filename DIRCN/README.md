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
    │   ├── base - BaseTrainer가 구현되어있음. 기본적인 function을 제외하고 구현해야할 부분을 abstract method로 남겨둠
    │   ├── config - json 형식의 config 파일을 업로드하여 runtime에서 args value를 확인할 수 있도록함.
    │   ├── dataset - dataset, dataloader를 관할하는 부분
    │   ├── fastmri - fastmri에서 주로 사용되는 coil_combine, fftc, math등을 구현함.
    │   ├── models - DIRCN을 구성하는 모듈을 low/high level에 따라 쪼개어 파일로 저장함, 다양한 ssim loss를 구현함.
    │   ├── preprocessing - data의 전처리를 담당하는 부분, kspace downsampling, image_cropping 등 data transform에서 활용됨.
    │   ├── trainer - one epoch 당 train, validation, test를 진행할 수 있도록 함수를 구현함. 특히 test phase에서는 reconstruction h5 파일을 result/DIRCN에 dump함
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
    │   └── best-validation
    │       ├── checkpoint-best.pth
    │       └── statistics.json
    ├── train.py 
    ├── recon.py - best-validation 모델을 활용해 leaderboard set의 reconstruction을 진행함
    ├── eval.py - reconstructed h5 파일과 leaderboard ground truth를 활용해서 masking을 활용한 leaderboard evaluation을 진행함.
    ├── dircn.json - train에 필요한 다양한 hyperparameters가 들어가있음.
    ├── README.md
    └── .gitignore
        
```

## 1. train/val/test
```
train data - [brain101.h5, ..., brain407.h5]: /root/input/train/image, /root/input/train/kspace
validation data - [brain1.h5, ..., brain100.h5]: /root/input/val/image, /root/input/val/kspace
test data - [brain_test1.h5, ..., brain_test58.h5]: /root/input/leaderboard/image, /root/input/leaderboard/kspace

# train, val set을 8:2로 분할하여 활용함.
```

## 2. how to start!(배포받은 서버기준)
```
# train(train data fitting 이후 validation data evaluation과 test data에 대한 reconstruction도 차례로 진행함)
cd /root/fastMRI/DIRCN/
python train.py

# reconstruction(test data에 대한 reconstruction)
cd /root/fastMRI/DIRCN/
python recon.py

# evaluate(dumping된 reconstruction h5파일에 대한 evaluation)
cd /root/fastMRI/DIRCN/
python eval.py

# reconstruction로 파일을 dump한 이후 evaluation을 진행하는 구조입니다.
```

## 3. training method
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
1. DIRCN에 관련해서 pretrained model을 찾지 못하여 위와 같이구성된 모델을 전체 주어진 set(train data, validation data)을 갖고 50 에포크를 training시켜 pretrained weight을 생성하였습니다. 이때는 단순히 train loss를 기준으로 goodness를 판단하였습니다.
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

## 4. our modification
![image](https://user-images.githubusercontent.com/39179946/185732142-44dcc3fb-d541-4b9d-bbc0-222c3e613780.png)
modification과 model에 관련해서는 차후 ppt에서 더욱 자세하게 설명하도록 하겠습니다.
