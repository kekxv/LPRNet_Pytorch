import torch.nn as nn
import torch

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

# --- 关键修改：自定义一个兼容 OpenCV 的 MaxPool 层 ---
class LPRPyTorchMaxPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(LPRPyTorchMaxPool, self).__init__()
        # 原代码 kernel_size=(1, 3, 3) -> 对应 2D kernel=(3, 3)
        self.kernel_size_2d = (kernel_size[1], kernel_size[2])

        # 原代码 stride=(depth_stride, height_stride, width_stride)
        self.stride_depth = stride[0]
        self.stride_2d = (stride[1], stride[2])

        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size_2d, stride=self.stride_2d)

    def forward(self, x):
        # 如果需要在 Channel 维度进行 Stride (例如 stride=(2,1,2))
        # 我们使用切片操作: x[:, ::2, :, :]
        # 这在 ONNX 中会被转换为 Slice 算子，OpenCV 完美支持
        if self.stride_depth > 1:
            x = x[:, ::self.stride_depth, :, :]

        return self.pool(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2

            # [修改] 替换 MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1))
            LPRPyTorchMaxPool(kernel_size=(1, 3, 3), stride=(1, 1, 1)), # 3

            small_basic_block(ch_in=64, ch_out=128),    # 4
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6

            # [修改] 替换 MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2))
            # 注意：这里 stride[0]=2，表示通道数减半，新类会自动处理
            LPRPyTorchMaxPool(kernel_size=(1, 3, 3), stride=(2, 1, 2)), # 7

            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # 11
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),

            # [修改] 替换 MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2))
            LPRPyTorchMaxPool(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14

            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # 22
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            # 注意：由于我们将 MaxPool3d 替换为了自定义层，层的索引并未改变
            # 原始索引：Conv(0), BN(1), ReLU(2), Pool(3), Block(4)...
            # keep_features 需要取的是 ReLU 之后的特征
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits

def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)
    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
