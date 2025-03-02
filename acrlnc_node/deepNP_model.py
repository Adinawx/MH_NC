import torch
import torch.nn as nn


class EstBlock(nn.Module):
    def __init__(self, input_size, hidden_size, out_features, threshold, device):
        super(EstBlock, self).__init__()
        self.th = threshold
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_features, device=device)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        s, _ = self.lstm(x)
        s = s[:, -1, :]
        s = self.fc(s)
        p = self.sig(s - self.th)
        ##########################
        out = torch.cat([p, s], dim=1)
        ##########################
        return out


# DNN LESS Memory: The DNN consists of rtt blocks, each contains an lstm layer followed by fc layer and sigmoid
# activation. The first block's input size is the memory_size, the others are memory_size+1 as they get the output of
# the previous layer.
class DeepNp(nn.Module):
    def __init__(self, input_size, hidden_size, future, threshold, device):
        super(DeepNp, self).__init__()
        self.future = future
        self.threshold = threshold
        self.BlockList = nn.ModuleList()
        for i in range(future):
            block = EstBlock(input_size=input_size, hidden_size=hidden_size, out_features=1, threshold=self.threshold,
                             device=device)
            self.BlockList.append(block)
        self.device = device

    def forward(self, x):

        # initialize
        max_mem = 500
        mem_vec = torch.zeros(x.shape[0], max_mem, device=self.device)
        x = x.to(self.device)
        all_pred = torch.zeros(x.shape[0], self.future, device=self.device)

        # first block
        out = self.BlockList[0](x).clone()
        pred_new = out[:, 0]
        sinr_pred = out[:, 1]
        all_pred[:, 0] = pred_new

        # rest of the blocks
        for i in range(1, self.future):

            if i <= max_mem:
                mem_vec[:, i - 1] = sinr_pred
                mem_vec_in = torch.unsqueeze(mem_vec[:, :i], dim=2)
            else:
                mem_vec[:, :-1] = mem_vec[:, 1:].clone()
                mem_vec[:, -1] = sinr_pred
                mem_vec_in = torch.unsqueeze(mem_vec, dim=2)

            input_vec = torch.cat([x, mem_vec_in], dim=1)
            out = self.BlockList[i](input_vec).clone()
            pred_new = out[:, 0]
            sinr_pred = out[:, 1]
            all_pred[:, i] = pred_new

        return all_pred



