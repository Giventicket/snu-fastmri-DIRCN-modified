# Weighted Sum DIRCN- 2022 fastMRI Challenge-

# Brave New World
Junpyo Seo, Joonwon Kang

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B80.png)

# Contents

Points of DIRCN

Our Modification

Whole Architecture

Training Strategy

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B81.png)

# Points of DIRCN

* Serial Connection of Refinement & Data Consistency Blocks
* Refinement module modification
  * Input Level Dense Connection
* U\-Net modification \(ResXUnet\)
  * Squeeze and Excitation CNN
  * Long Range Skip Connection

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B82.png)

<span style="color:#222222">Ottesen\, Jon Andre\, et al\. "A Densely Interconnected Network for Deep Learning Accelerated MRI\." </span>  <span style="color:#222222"> _arXiv_ </span>  <span style="color:#222222"> _ preprint arXiv:2207\.02073_ </span>  <span style="color:#222222"> \(2022\)\.</span>

# Serial Connection of R & DC Blocks

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B83.png)

<span style="color:#222222">Sriram\, </span>  <span style="color:#222222">Anuroop</span>  <span style="color:#222222">\, et al\. "End\-to\-end variational networks for accelerated MRI reconstruction\." </span>  <span style="color:#222222"> _International Conference on Medical Image Computing and Computer\-Assisted Intervention_ </span>  <span style="color:#222222">\. Springer\, Cham\, 2020\.</span>

# Input Level Dense Connection

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B84.png)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B85.png)

# U-Net modification (ResXUnet)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B86.png)

<span style="color:#FF0000">Basic</span>  <span style="color:#FF0000"> </span>  <span style="color:#FF0000">block\,</span>  <span style="color:#FF0000"> </span>  <span style="color:#FF0000">bottleneck</span>  <span style="color:#FF0000"> </span>  <span style="color:#FF0000">block \-> Squeeze and Excitation</span>

SILU\(Sigmoid Linear Unit\) activation

4 – depth block

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B87.png)

# U-Net modification (ResXUnet)Squeeze and Excitation CNN

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B88.png)

Squeeze: \(H\, W\, C\)  \(1\, 1\, C\)

Excitation: \(1\, 1\, C\)  \(1\, 1\, C\)

Rescaling: element\-wise multiplication between original tensor and attention weight

SE  __recalibrates__  channel\-wise feature responses by explicitly modelling  __interdependencies between channels__

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B89.png)

C’ = max\(1\, ratio \* C\)\, ratio = 1/16

Squeeze and excitation block\(SE\) implementation

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B810.png)

<span style="color:#7030A0">Conv1\,</span>

<span style="color:#7030A0">Norm1\,</span>

<span style="color:#7030A0">activation</span>

<span style="color:#E59EE2">Conv1\, norm1\,</span>

<span style="color:#E59EE2">activation</span>

<span style="color:#E59EE2">Conv2\, norm2\,</span>

<span style="color:#E59EE2">activation</span>

<span style="color:#7030A0"> __Basic block implementation__ </span>

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B811.png)

<span style="color:#E59EE2"> __bottleneck block implementation__ </span>

# U-Net modification (ResXUnet)Long Range Skip Connection

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B812.png)

long range skip\-connections will further improve gradient flow and further fine\-tune feature maps

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B813.png)

# Our Modification

* Gaussian SSIM Loss \+ L1 Loss
* Input Level Dense Connection
  * For the limited GPU memory\, hard to implement
  * Concatenation → Weighted Sum

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B814.png)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B815.png)

<span style="color:#222222">Liu\, </span>  <span style="color:#222222">Liqiang</span>  <span style="color:#222222">\, et al\. "Weighted Aggregating Feature Pyramid Network for Object Detection\." </span>  <span style="color:#222222"> _2020 International Conference on Computer Vision\, Image and Deep Learning \(CVIDL\)_ </span>  <span style="color:#222222">\. IEEE\, 2020\.</span>

# Whole Architecture

Reconstruct leaderboard

Images

Cropped

Input

\(C\, W\, W\)

Cropped

\(384\, 384\)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B816.png)

# Training Strategy

Mini Batch

Focal Loss

Learning Rate Adjustment

Transfer Learning

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B817.png)

# Mini Batch

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B818.png)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B819.png)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B820.png)

# Focal Loss

__Gt\-SSIM distribution of whole train set__

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B821.png)

Observation: Data imbalance problem

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B822.png)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B823.png)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B824.png)

# Learning Rate Adjustment

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B825.png)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B826.png)

# Transfer Learning

Method 1: Training additional blocks\, freezing original model

Cropped

Input

\(C\, W\, W\)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B827.png)

Method 2: Training SME block\, freezing Cascade Blocks

Cropped

Input

\(C\, W\, W\)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B828.png)

Method 3: Training Cascade Blocks\, freezing SME Blocks

Cropped

Input

\(C\, W\, W\)

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B829.png)

# Thank You!

![](img/Modified%20DIRCN%20for%20fastMRI_%EC%B5%9C%EC%A2%85%EB%B3%B830.png)

