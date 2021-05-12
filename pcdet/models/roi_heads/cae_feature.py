import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SpareNetEncode(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1
    output
    - feture:  one x feature_size
    """

    def __init__(
        self,
        bottleneck_size=2048,
        use_SElayer=True,
        hide_size=2048,
    ):
        super(SpareNetEncode, self).__init__()
        self.feat_extractor = EdgeConvResFeat(
            use_SElayer=use_SElayer, k=4, output_size=hide_size, hide_size=2048
        )
        self.linear = nn.Linear(hide_size, bottleneck_size)
        self.bn = nn.BatchNorm1d(bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def knn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices

    if torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx

def get_graph_feature(x, k: int = 20, idx=None):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    - idx: neighbor_idx
    outputs:
    - feature: b x npoints1 x (num_dims*2)
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class EdgeConvResFeat(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1
    output
    - feture:  b x feature_size
    """

    def __init__(
        self,
        num_point: int = 256,
        use_SElayer: bool = False,
        k: int = 4,
        hide_size: int = 2048,
        output_size: int = 2048,
    ):
        super(EdgeConvResFeat, self).__init__()
        self.use_SElayer = use_SElayer
        self.k = k
        self.hide_size = hide_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(133 * 2, self.hide_size // 16, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 8, kernel_size=1, bias=False
        )
        self.conv5 = nn.Conv1d(
            self.hide_size // 4, self.output_size // 2, kernel_size=1, bias=False
        )

        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)

        if use_SElayer:
            self.se1 = SELayer(channel=self.hide_size // 16)
            self.se2 = SELayer(channel=self.hide_size // 16)
            self.se3 = SELayer(channel=self.hide_size // 8)

        self.bn1 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn2 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn3 = nn.BatchNorm2d(self.hide_size // 8)
        self.bn5 = nn.BatchNorm1d(self.output_size // 2)

        self.resconv1 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.resconv2 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 8, kernel_size=1, bias=False
        )

    def forward(self, x):
        # x : [bs, 133, num_points]
        batch_size = x.size(0)
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)
            x = self.relu1(self.se1(self.bn1(self.conv1(x))))
            x1 = x.max(dim=-1, keepdim=False)[0]

            x2_res = self.resconv1(x1)
            x = get_graph_feature(x1, k=self.k)
            x = self.relu2(self.se2(self.bn2(self.conv2(x))))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.se3(self.bn3(self.conv3(x))))

        else:
            x = get_graph_feature(x, k=self.k)
            x = self.relu1(self.bn1(self.conv1(x)))
            x1 = x.max(dim=-1, keepdim=False)[0]

            x2_res = self.resconv1(x1)
            x = get_graph_feature(x1, k=self.k)
            x = self.relu2(self.bn2(self.conv2(x)))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.bn3(self.conv3(x)))

        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = x3 + x3_res

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu5(self.bn5(self.conv5(x)))

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]
        x = torch.cat((x1, x2), 1)  # [bs, 4096]

        x = x.view(-1, self.output_size)
        return x

class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)
    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)