import torch.nn as nn
import torch.nn.functional as F
import torch


class TemporalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, max_pool):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=(kernel))
        self.BN = nn.BatchNorm1d(output_dim)
        self.MP = nn.MaxPool1d(max_pool)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.BN(x))
        x = self.MP(x)

        return x


class SpatialBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, max_pool):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=(kernel))
        self.BN = nn.BatchNorm1d(output_dim)
        self.MP = nn.MaxPool1d(max_pool)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.BN(x))
        x = self.MP(x)

        return x


class FCBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.FC = nn.Linear(input_dim, output_dim)
        self.BN = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.FC(x)
        x = F.relu(self.BN(x))
        x = F.dropout(x)

        return x


class SexNet(nn.Module):

    def __init__(self,input_size=12):
        super().__init__()
        self.Temp1 = TemporalBlock(input_size, 16, 7, 2)
        self.Temp2 = TemporalBlock(16, 16, 5, 4)
        self.Temp3 = TemporalBlock(16, 32, 4, 2)
        self.Temp4 = TemporalBlock(32, 32, 5, 4)
        self.Temp5 = TemporalBlock(32, 64, 5, 2)
        self.Temp6 = TemporalBlock(64, 64, 3, 2)
        self.Temp7 = TemporalBlock(64, 64, 3, 2)
        self.Temp8 = TemporalBlock(64, 64, 3, 1)
        self.Spat = SpatialBlock(5, 128, 12, 2)
        self.FC1 = FCBlock(128 * 26, 128)
        self.FC2 = FCBlock(128, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.Temp1(x)
        x = self.Temp2(x)
        x = self.Temp3(x)
        x = self.Temp4(x)
        x = self.Temp5(x)
        x = self.Temp6(x)
        x = self.Temp7(x)
        x = self.Temp8(x)

        x = torch.swapaxes(x, 1, 2)
        x = self.Spat(x)
        x = x.view(x.shape[0], -1)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.out(x)

        return x

class AgeNet(nn.Module):
    def __init__(self,input_size=12):
        super().__init__()
        self.Temp1 = TemporalBlock(input_size,16,7,2)
        self.Temp2 = TemporalBlock(16,16,5,4)
        self.Temp3 = TemporalBlock(16,32,4,2)
        self.Temp4 = TemporalBlock(32,32,5,4)
        self.Temp5 = TemporalBlock(32,64,5,2)
        self.Temp6 = TemporalBlock(64,64,3,2)
        self.Temp7 = TemporalBlock(64,64,3,2)
        self.Temp8 = TemporalBlock(64,64,3,1)
        self.Spat = SpatialBlock(5,128,12,2)

        self.FC1 =  FCBlock(128*26,128)
        self.FC2 =  FCBlock(128,64)
        self.out = nn.Linear(64,1)

    def forward(self, x):
        x = self.Temp1(x)
        x = self.Temp2(x)
        x = self.Temp3(x)
        x = self.Temp4(x)
        x = self.Temp5(x)
        x = self.Temp6(x)
        x = self.Temp7(x)
        x = self.Temp8(x)
        x = torch.swapaxes(x, 1,2)
        x = self.Spat(x)
        x = x.view(x.shape[0],-1)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.out(x)

        return x.squeeze()