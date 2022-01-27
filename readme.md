## 实验环境

本次实验使用MistGPU平台租用的RTX3060运行

环境要求：

python

torch

torchvision

使用包：

```python
import torch
import torchvision
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
```

## 数据集下载

**mnist**：使用torchvision中的datasets，并下载

```python
import torchvision.datasets as datasets
mnist = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
```

## 运行方式

项目文件夹中共有五个python文件：

GAN.py

DCGAN.py

WGAN.py

WGAN_GP.py

graph_loss.py

分别在命令行中使用 python xxx.py即可运行

前四个是对抗生成网络

第五个是提取loss的csv文件，输出loss曲线图



dataset文件夹中存放了mnist数据集

img文件夹中存放了四种模型训练生成的图片和真实的图片

pth文件夹中存放了四种模型产生的生成器模型和判别器模型

loss文件夹中存放了四种模型生成器和判别器训练的损失，和损失的曲线图



## 实验结果

（img文件夹存放了训练结果，loss文件夹存放了损失图片，markdown文件展示图片需要图床，此处展示结论，详情可见实验报告和上述文件夹）

#### GAN

可以看到GAN网络可以基本生成可以辨识的图片，最开始训练出的图片数字比较同一，后续训练过后0-9的数字都可以生成，但是个别数字不清晰，模糊，看不出具体的数字形状，只能比较抽象的形态，说明GAN网络相应的生成效果并不算优秀。

而关于生成器和判别器的损失变化，两个对抗神经网络的损失变化是随机震荡的，这也说明了为什么GAN生成效果并不稳定。

#### DCGAN

可以看到DCGAN网络生成效果明显由于GAN网络生成的图片，最开始训练出的图片数字笔画比较混乱，无法形成明显的数字，后续训练过后0-9的数字都可以生成，相比GAN网络训练结果笔画清晰，结构明显，但是个别数字形状依旧比较混乱，特别是后期训练模型崩溃，所有损失变为0，生成器和判别器都失效了。

而关于生成器和判别器的损失变化，两个对抗神经网络的损失变化在正常训练阶段是有规律变化的，判别器损失一直比较小，而生成器损失震荡增加。

#### WGAN

可以看到WGAN网络生成效果总的来说不错，但未必由于DCGAN产生的图片，最开始训练出的图片数字有杂块，部分数字效果不错，后续训练效果逐渐变好，数字结构明显，但是个别部分表现不佳，后期训练出的图片数字基本能够显示，但个别笔画抽象，不如DCGAN中期表现。

而关于生成器和判别器的损失变化，两个对抗神经网络的损失变化都是在震荡减小，逐步收敛，这一点是之前GAN和DCGAN网络都没能做到的。

#### WGAN-GP

对比之前几个模型的生成效果，可以看到WGAN-GP网络生成效果是最优秀的，最开始训练出的图片数字初步成型，但数字细节模糊不清晰，后续训练效果逐渐变好，且多次训练过后结果仍旧稳定，和真实图片相差无几。

而关于生成器和判别器的损失变化，除了生成器初期阶段，两个对抗神经网络的损失变化都是在震荡减小，逐步收敛的，和WGAN训练情况类似，但优点在于训练效果更好，训练速度也稍微更快一些。
