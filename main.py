import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import Cityscapes
import torch.nn.functional as F
from torchvision import transforms
from onnx import shape_inference
from onnx import load,save
from torch.utils.data import DataLoader
#import cv2
import torchvision.transforms as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            y = y.float()
            y = torch.argmax(y, dim=1)
            y_hat = y_hat.float()
            l = loss(y_hat, y)
            total_loss += float(l)
            total_hits += sum(net(X).argmax(axis=1).type(y.dtype) == y)
            total_samples += y.numel()
    return float(total_loss) / len(data_iter), float(total_hits) / total_samples * 100


def train_epoch(net, train_iter, loss, optimizer, device):
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        X, y = X.to(device), y.to(device)
        y=torch.argmax(y.clone(), dim=1)
        y_hat = net(X)


        print("shape of image input:", X.size())
        print("shape of returned model:",y_hat.size())
        print("shape of expected image output",y.size())


        l = loss(y_hat, y)
        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss =total_loss+ float(l)
        total_hits =total_hits+ sum(y_hat.argmax(axis=1).type(y.dtype) == y)
        total_samples =total_samples+ y.numel()
    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples * 100


def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('Training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), weight_decay=0.0125,lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss, val_acc = evaluate_accuracy(net, val_iter, loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)
        print(
            f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}')
    test_loss, test_acc = evaluate_accuracy(net, test_iter, loss, device)
    print(f'Test loss {test_loss:.2f}, Test accuracy {test_acc:.2f}')

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels, 3,  stride, 1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else: return self.relu(out)


def trunc_normal_init(param,std, **kwargs):
    nn.init.trunc_normal_(param,std)


def constant_init(param,value,**kwargs):
    nn.init.constant_(param,value)


def kaiming_normal_init(param, **kwargs):
    nn.init.kaiming_normal_(param)


class MLP(nn.Module):

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels,hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels,out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8,
                 use_cross_kv=False):
        super().__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = nn.BatchNorm2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert self.same_in_out_chs, "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.k = torch.randn(inter_channels, in_channels, 1, 1)
            self.v = torch.randn(out_channels, inter_channels, 1, 1)

        self._init_weights(self)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, value=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, value=1.)
            constant_init(m.bias, value=.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, value=0.)

    def _act_sn(self, x):
        x1=x.clone().size()[0]
        x2=x.clone().size()[1]
        x3=x.clone().size()[2]
        x4=x.clone().size()[3]
        xf =( torch.reshape(x.clone(),( (x1*x2) // self.inter_channels, self.inter_channels, x3, x4))) * (self.inter_channels ** -0.5)
        xf = F.softmax(xf.clone(), dim=1)
        x1 = xf.clone().size()[0]
        x2 = xf.clone().size()[1]
        x3 = xf.clone().size()[2]
        x4 = xf.clone().size()[3]
        xF = torch.reshape(xf.clone(),(1, (x1*x2), x3, x4))

        return xF



    def _act_dn(self, x):

        x_shape = x.clone().size()
        y1 = x.clone().size()[0]
        y2 = x.clone().size()[1]
        y3 = x.clone().size()[2]
        y4 = x.clone().size()[3]
        h = x_shape[2]
        w = x_shape[3]
        xf = torch.reshape(x.clone(),(y1, self.num_heads, self.inter_channels // self.num_heads, (y1*y2*y3*y4)//(y1*self.num_heads*(self.inter_channels // self.num_heads))))
        xf = F.softmax(xf.clone(),dim=3)
        xf = xf.clone() / (torch.sum(xf.clone(), dim=2, keepdim=True) + 1e-06)

        y1 = xf.clone().size()[0]
        y2 = xf.clone().size()[1]
        y3 = xf.clone().size()[2]
        y4 = xf.clone().size()[3]
        xf = torch.reshape(xf.clone(),(y1, self.inter_channels, h, w))
        return xf

    def forward(self, x, cross_k=None, cross_v=None):

        x = self.norm(x)

        if not self.use_cross_kv:
            x = F.conv2d(
                x.clone(),
                self.k,
                bias=None,
                stride=2 if not self.same_in_out_chs else 1,
                padding=0)

            x = self._act_dn(x)
            x = F.conv2d(x.clone(), self.v, bias=None, stride=1,padding=0)
        else:
            assert (cross_k is not None) and (cross_v is not None), \
                "cross_k and cross_v should no be None when use_cross_kv"
            B = x.clone().size()[0]

            t1 = x.clone().size()[0]
            t2 = x.clone().size()[1]
            t3 = x.clone().size()[2]
            t4 = x.clone().size()[3]
            x = torch.reshape(x.clone(),(1, (t1*t2), t3, t4))
            x = F.conv2d(x.clone(), cross_k.clone(), bias=None, stride=1, padding=0,groups=B)
            x = self._act_sn(x.clone())
            x = F.conv2d(x.clone(), cross_v.clone(), bias=None, stride=1, padding=0,groups=B)
            t1 = x.clone().size()[0]
            t2 = x.clone().size()[1]
            t3 = x.clone().size()[2]
            t4 = x.clone().size()[3]
            x = torch.reshape(x.clone(),((t1*t2) // (self.in_channels), self.in_channels, t3, t4))
        return x



def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = torch.tensor(1 - drop_prob)
    shape = (x.clone().size()[0],) + (1,) * (x.clone().ndimension() - 1)
    random_tensor = keep_prob + torch.rand(shape).type(dtype=type(x))
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.clone().divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class EABlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 use_cross_kv=True,
                 cross_size=12):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        # low resolution
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                nn.BatchNorm2d(in_channels_l),
                nn.Conv2d(in_channels_l,out_channels_l, 1, 2, 0))
            self.attn_shortcut_l.apply(self._init_weights_kaiming)
        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=out_channels_l,
            num_heads=num_heads,
            use_cross_kv=False)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else Identity()

        # compression
        self.compression = nn.Sequential(
            nn.BatchNorm2d(out_channels_l),
            nn.ReLU(),
            nn.Conv2d(out_channels_l,out_channels_h, kernel_size=1))
        self.compression.apply(self._init_weights_kaiming)

        # high resolution
        self.attn_h = ExternalAttention(
            in_channels_h,
            in_channels_h,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv)
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate)
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                nn.BatchNorm2d(out_channels_l),
                nn.AdaptiveMaxPool2d(output_size=(self.cross_size,self.cross_size)),
                nn.Conv2d(out_channels_l, 2 * out_channels_h, 1, 1, 0))
            self.cross_kv.apply(self._init_weights)

        # injection
        if use_injection:
            self.down = nn.Sequential(
                nn.BatchNorm2d(out_channels_h),
                nn.ReLU(),
                nn.Conv2d(out_channels_h,out_channels_l // 2,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(out_channels_l//2),
                nn.ReLU(),
                nn.Conv2d(out_channels_l // 2,out_channels_l,kernel_size=3,stride=2,padding=1), )
            self.down.apply(self._init_weights_kaiming)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def _init_weights_kaiming(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def forward(self, x):
        x_h,x_l=x

        # low resolution
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))

        # compression
        x_h_shape = x_h.size()[2:]
        x_l_cp = self.compression(x_l.clone())
        x_h =x_h + F.interpolate(x_l_cp, size=x_h_shape, mode='bilinear')

        # high resolution
        if not self.use_cross_kv:
            x_h = x_h.clone() + self.drop_path(self.attn_h(x_h.clone()))
        else:
            cross_kv = self.cross_kv(x_l.clone())  # n,2*out_channels_h,12,12

            a1 = cross_kv.clone().size()[0]
            a2 = cross_kv.clone().size()[1]
            a3 = cross_kv.clone().size()[2]
            a4 = cross_kv.clone().size()[3]

            cross_k =torch.empty(a1, a2//2, a3, a4)
            cross_v = torch.empty(a1, a2 // 2, a3, a4)

            a1 = cross_k.clone().size()[0]
            a2 = cross_k.clone().size()[1]
            a3 = cross_k.clone().size()[2]
            a4 = cross_k.clone().size()[3]
            cross_k=torch.reshape(cross_k.clone(), (a1, a3, a4, a2))

            a1 = cross_k.clone().size()[0]
            a2 = cross_k.clone().size()[1]
            a3 = cross_k.clone().size()[2]
            a4 = cross_k.clone().size()[3]
            cross_k = torch.reshape(cross_k.clone(), ((a1*a2*a3*a4)//self.out_channels_h, self.out_channels_h, 1, 1))

            a1 = cross_v.clone().size()[0]
            a2 = cross_v.clone().size()[1]
            a3 = cross_v.clone().size()[2]
            a4 = cross_v.clone().size()[3]
            cross_v = torch.reshape(cross_v.clone(),((a1*a2*a3*a4)//(self.cross_size * self.cross_size), self.cross_size * self.cross_size, 1,1))
            x_h = x_h.clone() + self.drop_path(self.attn_h(x_h.clone(), cross_k.clone(),cross_v.clone()))

        x_h = x_h.clone() + self.drop_path(self.mlp_h(x_h.clone()))

        # injection
        if self.use_injection:
            x_l = x_l.clone() + self.down(x_h.clone())

        return x_h, x_l


class DAPPM(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, lr_mult):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,inter_channels, kernel_size=1))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,inter_channels, kernel_size=1))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,inter_channels, kernel_size=1))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), #
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,inter_channels, kernel_size=1))
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,inter_channels, kernel_size=1))

        self.process1 = nn.Sequential(nn.BatchNorm2d(inter_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(inter_channels,inter_channels, kernel_size=3, padding=1))
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels,inter_channels, kernel_size=3, padding=1))
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels,inter_channels, kernel_size=3, padding=1))
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels,inter_channels, kernel_size=3, padding=1))

        self.compression = nn.Sequential(
            nn.BatchNorm2d(inter_channels*5),
            nn.ReLU(),
            nn.Conv2d(inter_channels*5,out_channels, kernel_size=1))
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels, kernel_size=1))

    def forward(self, x):

        x_shape = x.size()[2:]

        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x), size=x_shape, mode='bilinear') + x_list[0].clone())))
        x_list.append((self.process2((F.interpolate(self.scale2(x), size=x_shape, mode='bilinear') + x_list[1].clone()))))
        x_list.append(self.process3((F.interpolate(self.scale3(x), size=x_shape, mode='bilinear') + x_list[2].clone())))
        x_list.append(self.process4((F.interpolate(self.scale4(x), size=x_shape, mode='bilinear') + x_list[3].clone())))

        out = self.compression(torch.concat(x_list,dim=1)) + self.shortcut(x)
        return out


class RTFormer(nn.Module):

    def __init__(self,
                 num_classes=19,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=128,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 use_aux_head=True,
                 use_injection=[True, True],
                 lr_mult=10.,
                 cross_size=12,
                 in_channels=3,
                 pretrained=None):
        super().__init__()
        self.base_channels = base_channels
        base_chs = base_channels


#STEM
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(),
            nn.Conv2d(base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(), )
        self.relu = nn.ReLU()

#STAGES 1-3
        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,layer_nums[0])
        self.layer2 = self._make_layer(BasicBlock, base_chs, base_chs*2 , layer_nums[1])
        self.layer3 = self._make_layer(BasicBlock, base_chs*2 , base_chs*4 , layer_nums[2], stride=2)
        self.layer3_ =self._make_layer(BasicBlock, base_chs*2 , base_chs*2 , 1)
        self.compression3 = nn.Sequential(
            nn.BatchNorm2d(base_chs*4),
            nn.ReLU(),
            nn.Conv2d(base_chs*4,base_chs * 2, kernel_size=1),)
#TRANSFORMER
        self.layer4 = EABlock(
            in_channels=[base_chs * 2, base_chs * 4],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[0],
            use_cross_kv=True,
            cross_size=cross_size)

        self.layer5 = EABlock(
            in_channels=[base_chs * 2, base_chs * 8],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[1],
            use_cross_kv=True,
            cross_size=cross_size)

#HEAD
        self.spp = DAPPM(base_chs * 8, spp_channels, base_chs * 2, lr_mult=lr_mult)
        self.seghead = SegHead(base_chs * 4, int(head_channels * 2), num_classes, lr_mult=lr_mult)
        self.use_aux_head = use_aux_head
        if self.use_aux_head:
            self.seghead_extra = SegHead(base_chs * 2, head_channels, num_classes, lr_mult=lr_mult)

        self.pretrained = pretrained


    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))

        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def init_weight(self):
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.layer3_.apply(self._init_weights_kaiming)
        self.compression3.apply(self._init_weights_kaiming)
        self.spp.apply(self._init_weights_kaiming)
        self.seghead.apply(self._init_weights_kaiming)
        if self.use_aux_head:
            self.seghead_extra.apply(self._init_weights_kaiming)


    def forward(self, x):
        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3_ = x2 + F.interpolate(self.compression3(x3), size=(x2.clone()).size()[2:], mode='bilinear')
        x3_ = self.layer3_(self.relu(x3_))  # 2c, 1/8
        x4_, x4 = self.layer4([self.relu(x3_), self.relu(x3)])  # 2c, 1/8; 8c, 1/16

        x5_, x5 = self.layer5([self.relu(x4_), self.relu(x4)])  # 2c, 1/8; 8c, 1/32

        x6 = self.spp(x5.clone())

        x6 = F.interpolate(x6, size=x5_.size()[2:], mode='bilinear')  # 2c, 1/8
        x_out = self.seghead(torch.concat([x5_, x6], dim=1))  # 4c, 1/8

        logit_list = [x_out]

        if self.use_aux_head:
            x_out_extra = self.seghead_extra(x3_)
            logit_list.append(x_out_extra)

        logit_list = [
            F.interpolate(
                logit,
                x.size()[2:],
                mode='bilinear',
                align_corners=False) for logit in logit_list
        ]

        return logit_list[0]



class SegHead(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, lr_mult):
        self.init__ = super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,inter_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(inter_channels,out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out


if __name__ == '__main__':

      net=RTFormer()

      t = torch.randn(3, 3, 512, 1024)
      net(t)

      print(net)

      num_epochs, lr, batch_size = 1, 0.0004, 3

      train_iter = torch.utils.data.DataLoader(Cityscapes(r'C:\Users\Tania\Desktop\Cityscapes', split='train', mode='fine', target_type='semantic',transform=transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Resize(512),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),target_transform=transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Resize(512)])),batch_size=3)
      val_iter = torch.utils.data.DataLoader(Cityscapes(r'C:\Users\Tania\Desktop\Cityscapes', split='val', mode='fine', target_type='semantic',transform=transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Resize(512),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),target_transform=transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Resize(512)])),batch_size=3)
      test_iter = torch.utils.data.DataLoader(Cityscapes(r'C:\Users\Tania\Desktop\Cityscapes', split='test', mode='fine', target_type='semantic',transform=transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Resize(512),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),target_transform=transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Resize(512)])),batch_size=3)




      net.eval()
      train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(net, train_iter, val_iter, test_iter, num_epochs, lr,try_gpu())


