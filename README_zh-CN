# 使用U-Net和GAN教程进行图像着色


**如果您已经阅读了说明，可以直接转到以heading:_1 - 实现 我们的基准开头的代码_**

![title image](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/main.png?raw=1)
深度学习最令人兴奋的应用之一是对黑白图像进行着色。几年前，这项任务需要大量的人工输入和硬编码，但现在，借助人工智能和深度学习的力量，整个过程可以端到端完成。你可能认为你需要大量的数据或很长的训练时间来从头开始训练你的模型来完成这项任务，但在过去的几周里，我做了这项工作，尝试了许多不同的模型架构、损失函数、训练策略等，最后开发了一个有效的策略来训练这样的模型，在相当小的数据集上，训练时间非常短。在这篇文章中，我将解释我是如何做到这一点的，包括代码！，以及有帮助的策略和没有帮助的策略。在此之前，我将解释着色问题，并简要回顾一下近年来所做的工作。在本文的其余部分中，我假设您已经掌握了深度学习、GAN和PyTorch库的基本知识。开始吧！


## 着色问题简介

在这里，我将给你一些基本知识，你可能需要了解以下代码中的模型的作用。

### RGB与L\*a\*b

正如你可能知道的，当我们加载一个图像时，我们会得到一个3级（高度、宽度、颜色）数组，最后一个轴包含我们图像的颜色数据。这些数据在RGB颜色空间中表示颜色，每个像素有3个数字，表示该像素的红、绿、蓝程度。在下面的图像中，你可以看到在 "主图像 "的左边部分（最左边的图像），我们有蓝色的颜色，所以在图像的蓝色通道中，那一部分的数值较高，变成了暗色。

![rgb image](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/rgb.jpg?raw=1)

在L\*a\*b色彩空间中，我们对每个像素又有三个数字，但这些数字有不同的含义。第一个数字（通道），L，编码每个像素的亮度，当我们将这个通道可视化时（下面一排的第二张图片），它显示为一个黑白图像。频道*a和*b分别编码每个像素的绿-红和黄-蓝的程度。在下面的图片中，你可以分别看到L\*a\*b色彩空间的每个通道。

![lab image](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/lab.jpg?raw=1)

在我研究的所有论文和我在GitHub上查看的所有关于着色的代码中，人们使用L\*a\*b颜色空间而不是RGB来训练模型。这种选择有几个原因，但我要给你一个直观的印象，即我们为什么要做这种选择。为了训练一个用于着色的模型，我们应该给它一个灰度图像，并希望它能把它变成彩色的。当使用L\*a\*b时，我们可以把L通道交给模型（也就是灰度图像），并希望它预测其他两个通道（\*a, \*b），在它预测之后，我们把所有的通道连接起来，我们就得到了彩色的图像。但是，如果你使用RGB，你必须首先将你的图像转换成灰度，将灰度图像提供给模型，并希望它能为你预测3个数字，这是一个更困难和不稳定的任务，因为3个数字的可能组合比两个数字多。如果我们假设每个数字有256种选择（在8位无符号整数图像中，这是真正的选择数），预测每个像素的3个数字是在256³种组合中选择，这超过了1600万种选择，但当预测两个数字时，我们有大约65000种选择（实际上，我们不会像分类任务那样疯狂地选择这些数字，我写这些数字只是为了给你一个直觉）。

## 如何解决这个问题

在过去的几年里，人们提出了许多不同的解决方案，通过使用深度学习对图像进行着色。[_**彩色图像着色**_](https://arxiv.org/abs/1603.08511)的论文将这个问题作为分类任务来处理，他们还考虑了这个问题的不确定性（例如，图像中的汽车可以呈现出许多不同的有效颜色，我们无法确定它的任何颜色）；然而，另一篇论文将这个问题作为回归任务来处理（还做了一些调整！）。每种方法都有其优点和缺点，但在本文中，我们将使用一种不同的策略。

### 我们要使用的策略

[_**条件对抗网络的图像到图像翻译**_](https://arxiv.org/abs/1611.07004)论文，你可能知道它的名字pix2pix，提出了深度学习中许多图像到图像任务的一般解决方案，其中之一是着色。在这种方法中，使用了两种损失。L1损失，这使得它成为一项回归任务，以及对抗性（GAN）损失，这有助于以无监督的方式解决问题（通过给输出分配一个数字，表明它们看起来有多 "真实"！）。

在本教程中，我将首先实现作者在论文中的做法，然后我将介绍一个全新的生成器模型和训练策略的一些调整，这将大大有助于减少所需数据集的大小，同时获得惊人的结果。请继续关注 :)

###对GAN世界的深入研究

如前所述，我们将建立一个GAN（具体来说是一个条件GAN），并使用一个额外的损失函数，L1损失。让我们从GAN开始。

正如你可能知道的，在GAN中，我们有一个生成器和一个判别器模型，它们一起学习解决一个问题。在我们的设定中，生成器模型需要一个灰度图像（单通道图像），并产生一个双通道图像，一个通道为 \*a ，另一个为 \*b 。鉴别器将这两个产生的通道与输入的灰度图像连接起来，并决定这个新的3通道图像是假的还是真的。当然，鉴别器也需要看到一些真实的图像（又是Lab色彩空间中的3通道图像），这些图像不是由生成器产生的，应该知道它们是真的。 

那么，我们提到的 "条件 "是什么？那么，生成器和鉴别器都看到的灰度图像是我们在GAN中提供给两个模型的条件，并期望它们考虑到这个条件。

让我们来看看这个数学模型。把_**x**_看作是灰度图像，_**z**_看作是发生器的输入噪声，_**y**_看作是我们希望从发生器得到的2通道输出（它也可以代表真实图像的2个彩色通道）。另外，_**G**_是发生器模型，_**D**_是判别器。那么我们的条件GAN的损失将是。

![GAN损失](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/GAN_loss.jpg?raw=1)

请注意，_***x**_是给两个模型的，这是我们引入这个游戏的两个玩家的条件。实际上，我们并不是像你所期望的那样给生成器输入一个 "n "维的随机噪声向量，而是在生成器的结构中以丢弃层的形式引入噪声（有一些很酷的东西，你会在文章的最后一节看到）。

### 我们优化的损失函数

先前的损失函数有助于产生好看的彩色图像，看起来很真实，但为了进一步帮助模型，并在我们的任务中引入一些监督，我们将这个损失函数与预测颜色与实际颜色的L1损失（你可能知道L1损失是平均绝对误差）相结合。

![L1损失](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/l1_loss.jpg?raw=1)

如果我们单独使用L1损失，模型仍然会学习对图像进行着色，但它会比较保守，大多数时候会使用 "灰色 "或 "棕色 "这样的颜色，因为当它怀疑哪种颜色是最好的时候，它会取平均值并使用这些颜色来尽可能地减少L1损失（这与超级分辨率任务中L1或L2损失的模糊效果类似）。另外，L1损失比L2损失（或平均平方误差）更受欢迎，因为它减少了产生偏灰的图像的那种影响。因此，我们的综合损失函数将是。

![损失](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/loss.jpg?raw=1)

其中_**λ**_是一个系数，用于平衡两种损失对最终损失的贡献（当然，判别器损失不涉及L1损失）。

好了，我想理论上已经足够了! 让我们动手写写代码吧! 在下面的章节中，**我首先介绍了实现论文的代码，**在之后的章节中，**我将介绍一个更好的策略，在一两个小时的训练中获得真正惊人的结果，而且不需要大量的数据！**。

## 1 - 实施文件 - 我们的基线

### 1.1-加载图像路径

论文中使用了整个ImageNet数据集（有130万张图片！），但在这里我只使用了COCO数据集的8000张图片进行训练，我的设备上有这些图片。因此，我们的训练集大小是论文中使用的0.6%!
你几乎可以使用任何数据集来完成这项任务，只要它包含许多不同的场景和地点，你希望它能学会着色。例如，你可以使用ImageNet，但在这个项目中你只需要它的8000张图片。
#%%
import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None

### 1.1.x 为运行代码准备Colab

如果你是在**谷歌Colab上打开的，你可以解压缩并运行以下内容来安装fastai。本教程中几乎所有的代码都是用**纯PyTorch**的。我们在这里只需要fastai来下载COCO数据集的一部分，以及在本教程第二部分的另一个步骤中。

此外，请确保将运行时间设置为**GPU**，以便能够更快地训练模型。
#%%
#!pip install fastai==2.4

下面将从COCO数据集中下载大约20000张图片。注意，**我们将只使用其中的8000张**进行训练。你也可以使用任何其他数据集，如ImageNet，只要它包含各种场景和地点。
#%%
# from fastai.data.external import untar_data, URLs
# coco_path = untar_data(URLs.COCO_SAMPLE)
# coco_path = str(coco_path) + "/train_sample"
# use_colab = True
#%%
if use_colab == True:
    path = coco_path
else:
    path = "Your path to the dataset"
    
paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))
#%%
_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

虽然我们使用的是相同的数据集和训练样本数量，但你训练模型的确切的8000张图片可能会有所不同（虽然我们是在播种！），因为这里的数据集只有20000张不同排序的图片，而我从完整的数据集中抽出了10000张图片。

### 1.2- 制作数据集和数据加载器

我希望这段代码是不言自明的。我正在调整图像的大小并进行水平翻转（只有在训练集时才翻转），然后我读取RGB图像，将其转换为Lab色彩空间，并将第一（灰度）通道和彩色通道分开，分别作为我的输入和模型的目标。然后我就开始制作数据加载器。
#%%
SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
#%%
train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

### 1.3- 论文提出的 Generator proposed

这个有点复杂，需要解释。这段代码实现了一个U型网，作为我们GAN的生成器。代码的细节超出了本文的范围，但需要理解的是，它从中间部分（在U型中向下）制作U型网，并在每次迭代时分别在中间模块的左边和右边增加向下采样和向上采样的模块，直到达到输入模块和输出模块。请看下面这张图片，它是我根据文章中的一张图片制作的，让你更好地了解代码中发生的情况。

![unet](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/unet.png?raw=1)

蓝色的矩形显示了相关模块与代码的构建顺序。我们将建立的U-Net比这张图中所描述的有更多的层次，但这足以让你明白这个概念。在代码中还注意到，我们要往下走8层，所以如果我们从256乘256的图像开始，在U-Net的中间，我们将得到一个1乘1（256/2⁸）的图像，然后它被向上采样，产生一个256乘256的图像（有两个通道）。这个代码片段真的很令人兴奋，我强烈建议你玩一玩，以完全掌握它的每一行在做什么。
#%%
class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)

### 1.4- 判别器

我们的鉴别器的结构是相当直接的。这段代码通过堆叠Conv-BatchNorm-LeackyReLU的块来实现一个模型，以决定输入图像是假还是真的。请注意，第一个和最后一个块不使用归一化，最后一个块没有激活函数（它被嵌入到我们将使用的损失函数中）。
#%%
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

让我们来看看它的区块。
#%%
PatchDiscriminator(3)

以及它的输出形状。
#%%
discriminator = PatchDiscriminator(3)
dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size
out = discriminator(dummy_input)
out.shape

我们在这里使用的是一个 "补丁 "鉴别器。好吧，它是什么？在一个普通的鉴别器中，模型输出一个数字（一个标尺），代表模型认为输入（也就是整个图像）的真实程度（或假的）。在斑块鉴别器中，模型为输入图像的每一个70×70像素的斑块输出一个数字，并为每一个斑块分别决定它是否是假的。在我看来，将这样的模型用于着色任务是合理的，因为模型需要做的局部变化真的很重要，也许像香草鉴别器那样对整个图像进行决定不能照顾到这项任务的微妙之处。这里，模型的输出形状是30乘30，但这并不意味着我们的补丁是30乘30。当你计算这900个（30乘以30）输出数字中的每一个的感受野时，就会得到实际的补丁大小，在我们的例子中，这将是70乘以70。

### 1.5- GAN损失

这是一个方便的类，我们可以用来计算我们最终模型的GAN损失。在__init__中，我们决定使用哪种损失（在我们的项目中是 "vanilla"），并注册一些常数张量作为 "真 "和 "假 "标签。然后，当我们调用这个模块时，它将生成一个适当的充满零或一的张量（根据我们在这个阶段的需要）并计算损失。
#%%
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

### 1.x模型初始化

在TowardsDataScince的文章中，我并没有解释这个函数。下面是我们初始化模型的逻辑。我们将用0.0的平均值和0.02的标准差来初始化我们的模型的权重，这就是文章中提出的超参数。
#%%
def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

### 1.6- 将所有东西放在一起

这个类汇集了之前所有的部分，并实现了一些方法来处理训练我们的完整模型。让我们来研究一下它。 

在__init__中，我们使用之前定义的函数和类来定义我们的生成器和判别器，我们也用init_model函数来初始化它们，我在这里没有解释，但你可以参考我的GitHub仓库来看看它是如何工作的。然后我们定义我们的两个损失函数以及生成器和判别器的优化器。 

整个工作都在这个类的优化方法中完成。首先，每一次迭代（一批训练集）我们都要调用模块的前进方法，并将输出存储在类的fake_color变量中。 

然后，我们首先使用backward_D方法训练判别器，将生成器生成的假图像送入判别器（确保将它们从生成器的图形中分离出来，以便它们作为判别器的常量，像正常图像一样），并将它们标记为假图像。然后，我们从训练集中拿出一批真实的图像给鉴别器，并将它们标记为真实的。我们将假的和真的两个损失相加，取其平均值，然后将最终的损失向后调用。 
现在，我们可以训练生成器了。在逆向_G方法中，我们向鉴别器提供假图像，并试图通过给它们贴上真实的标签来欺骗它，并计算出对抗性损失。正如我前面提到的，我们也使用L1损失，计算预测的两个通道和目标的两个通道之间的距离，并将这个损失乘以一个系数（在我们的例子中是100）来平衡这两个损失，然后将这个损失加到对抗性损失中。然后我们称之为损失的后退法。
#%%
class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

### 1.xx 工具函数

这些函数也没有包括在TDS文章的解释中。这些只是一些实用的函数，用来记录我们网络的损失，也可以在训练过程中直观地看到结果。所以在这里你可以查看它们。
#%%
class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

### 1.7- 训练功能

我希望这个代码是不言自明的。在像Nvidia P5000这样并不强大的GPU上，每个纪元需要4分钟。因此，如果你使用1080Ti或更高版本，它将会快很多。
#%%
def train_model(model, train_dl, epochs, display_every=200):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs

model = MainModel()
train_model(model, train_dl, 100)

在Colab上，每次历时大约需要3到4分钟。经过大约20次，你应该看到一些合理的结果。

好吧，我让模型再训练一段时间（大约100个epochs）。下面是我们的基线模型的结果。

![基线](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/baseline.png?raw=1)

正如你所看到的，尽管这个基线模型对图像中一些最常见的物体如天空、树木等有一些基本的了解，但它的输出远远没有达到吸引人的程度，它不能决定稀有物体的颜色。它还显示了一些颜色的溢出和圆圈状的颜色块（第二行第一张图片的中心），这一点都不好。因此，在这个小数据集上，我们似乎不能用这个策略获得好的结果。**因此，我们改变了我们的策略！**。

##2- 一个新的策略--最终模式

这里是本文的重点，我将在这里解释我是如何克服最后提到的问题的。受超级分辨率文献中的一个想法的启发，我决定以有监督的和确定的方式分别对生成器进行预训练，以避免GAN游戏中的 "盲人摸象 "问题，即生成器和鉴别器在训练开始时对任务一无所知。 

事实上，我在两个阶段使用预训练。1- 生成器的骨干（向下采样路径）是一个预训练过的分类模型（在ImageNet上）2- 整个生成器将被预训练过的任务是L1损失的着色。

事实上，我将使用经过预训练的ResNet18作为我的U-Net的骨干，为了完成第二阶段的预训练，我们将在我们的训练集上只用L1损失来训练U-Net。然后，我们将转向对抗性和L1损失的组合，正如我们在上一节所做的那样。

### 2.1- 使用新的生成器

使用ResNet主干构建U-Net并非易事，因此我将使用fastai库的动态U-Net模块轻松构建一个。您可以简单地使用pip或conda安装fastai（如果您在本教程开始时还没有）。这是[文档]的链接(https://docs.fast.ai/).
####2022年1月8日更新：<br>
您需要安装fastai 2.4版，以便运行以下行代码，而不会出现错误。
如果您已经在教程开始时使用该单元安装了它，则无需在此处再次安装。
<br><br><br>
#%%
# pip install fastai==2.4
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
#%%
def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

这就是了! 只需这几行代码，你就可以轻松建立这样一个复杂的模型。create_body函数加载ResNet18架构的预训练权重，并切割模型以移除最后两层（GlobalAveragePooling和一个用于ImageNet分类任务的线性层）。然后，DynamicUnet使用这个骨架建立一个具有所需输出通道（在我们的例子中是2个）、输入大小为256的U-Net。

### 2.2 为着色任务预训练生成器
#%%
def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

net_G = build_res_unet(n_input=1, n_output=2, size=256)
opt = optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()        
pretrain_generator(net_G, train_dl, opt, criterion, 20)
#torch.save(net_G.state_dict(), "res18-unet.pt")

通过这个简单的函数，我们对生成器进行20个epochs的预训练，然后保存其权重。这在Colab上需要一个小时。在下一节中，我们将使用这个模型作为GAN的生成器，并像以前一样训练整个网络。

### 2.3 把所有东西放在一起，再来一次!

如果你想自己训练模型，运行下面的单元。相反，如果你想使用预训练的权重，跳过这个单元，运行后面的单元。
#%%
net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
train_model(model, train_dl, 20)

在这里，我首先为生成器加载保存的权重（你在上一节已经保存了），然后我在我们的MainModel类中使用这个模型作为生成器，这样可以防止它随机初始化生成器。然后我们训练这个模型10到20个epochs! (与上一节中不使用预训练时的100个epochs相比）。在Colab上，每个epoch大约需要3到4分钟。

如果你在Colab上并且想使用预训练的权重，运行以下单元，从我的google drive上下载权重并加载到模型上。
#%%
# !gdown --id 1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV
#%%
# net_G = build_res_unet(n_input=1, n_output=2, size=256)
# net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
# model = MainModel(net_G=net_G)
# model.load_state_dict(torch.load("final_model_weights.pt", map_location=device))

现在，我将展示这个最终模型在测试集（它在训练期间从未见过的黑白图像）上的结果，包括本文最开始的主标题图像。

![输出1](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/main.png?raw=1)
左图：来自测试集的黑白图像输入；右图：本教程最终模型的彩色化输出。
---
![output2](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/img1.png?raw=1)
左图：从测试集输入的黑白图像|右图：本教程最终模型的着色输出结果
---
![output3](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/img2.png?raw=1)
左图：从测试集输入的黑白图像，右图：本教程最终模型的彩色输出。
---

## 一个意外的发现。你可以安全地删除Dropout!

记得我在本文开头解释条件GAN的理论时，曾说过论文作者提出的生成器架构中的噪声来源是辍学层。然而，当我调查了我们在fastai的帮助下建立的U-Net时，我并没有发现其中有任何dropout层的存在。事实上，我首先训练了最终的模型并得到了结果，然后我调查了发生器并发现了这一点。 

那么，对抗性训练是不是没有用？如果没有噪音，发生器怎么可能对输出产生创造性的影响？是否有可能输入到生成器的灰度图像也起到了噪声的作用？这些都是我当时的确切问题。 

因此，我决定给菲利普-伊索拉博士发电子邮件，他是我们在这里实施的同一篇论文的第一作者，他亲切地回答了这些问题。根据他所说的，这个有条件的GAN仍然可以在没有辍学的情况下工作，但由于缺乏那个噪音，输出将更加确定；然而，在那个输入的灰度图像中仍然有足够的信息，使发生器能够产生令人信服的输出。
实际上，我在实践中看到，对抗性训练确实很有帮助。在下一节也是最后一节，我将比较没有经过对抗性训练的U-Net的结果和经过对抗性训练的最终输出。

## 比较经过预训练的U-Net和没有经过对抗性训练的结果

我在实验中发现的一个很酷的事情是，我们用ResNet18骨干构建的U-Net在仅用L1 Loss进行预训练后（最后的对抗训练前的一个步骤），在图像着色方面已经很出色了。但是，这个模型仍然很保守，当它不确定物体是什么或者它应该是什么颜色的时候，它鼓励使用偏灰的颜色。然而，对于图像中常见的场景，如天空、树木、草地等，它的表现确实很好。 

在这里，我向你展示了没有对抗性训练的U-Net和有对抗性训练的U-Net的输出结果，以更好地描述对抗性训练在我们的案例中所带来的显著差异。

![比较](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/comparison1.png?raw=1)
(左图：未经对抗性训练的预训练U-Net | 右图：经对抗性训练的预训练U-Net)
---

你也可以看看下面的GIF，以更好地观察图像之间的差异。

![anim](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/anim_compare.gif?raw=1)
(最后两张图片的动画，以更好地看到对抗性训练所带来的显著差异)
---

## 最后的话

这个项目对我来说充满了重要的教训。在上个月，我花了很多时间来实现很多不同的论文，每篇论文都有不同的策略，我花了相当长的时间，在经历了大量的失败之后，才想出了这个训练方法。现在你可以看到，对发电机进行预训练对模型有很大帮助，并改善了结果。

我还了解到，有些观察结果虽然一开始觉得是你的一个糟糕的错误，但却值得关注和进一步调查；比如这个项目中的辍学案例。感谢深度学习和人工智能的有益社区，你可以很容易地询问专家并得到你需要的答案，对你刚才的猜测变得更加自信。 

我要感谢这篇精彩论文的作者们的出色工作，也要感谢[这篇论文的伟大的GitHub仓库](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)，我从中借用了一些代码（经过修改和简化）。我真的很喜欢计算机科学和人工智能的社区，喜欢他们为改善这个领域而做的所有努力，也喜欢他们的贡献为所有人所用。我很高兴能成为这个社区的一个小部分。
