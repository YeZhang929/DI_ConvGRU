import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size, hidden_dim):
        return Variable(torch.zeros(batch_size, hidden_dim, self.height, self.width)).cuda()

    def forward(self, input_tensor, h_cur):

        #print(input_tensor.shape,h_cur.shape)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class DI_ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(DI_ConvGRU, self).__init__()
        self.height, self.width = input_size
        self.input_dim = [input_dim] + hidden_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self._all_layers = []

        for i in range(0, self.num_layers):
            name = 'cell{}'.format(i)
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell = ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size,
                                         bias=self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, inputs, y_obs_all):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            inputs = inputs.permute(1, 0, 2, 3, 4).cuda()
            y_obs_all = y_obs_all.permute(1, 0, 2, 3, 4).cuda()
        batch_size, NT, Nf, Nlon, Nlat = inputs.shape

        layer_output_list = []
        last_state_list = []
        h0 = torch.zeros(batch_size, 1, Nlon, Nlat).cuda()
        # h0 = torch.FloatTensor(batch_size, 1, Nlon, Nlat).fill_(0.2064).cuda()
        internal_state = []
        output_inner = []
        for t in range(NT):
            y_obs = y_obs_all[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                name = 'cell{}'.format(layer_idx)
                if t == 0:
                    bsize, _, height, width = y_obs.size()
                    h = getattr(self, name).init_hidden(bsize, self.hidden_dim[layer_idx])
                    internal_state.append(h)
                    if layer_idx == 0:
                        h = internal_state[layer_idx]
                        mask = y_obs == y_obs
                        x1 = h0
                        x1[mask] = y_obs[mask]
                        input = x1
                        xt = torch.cat((inputs[:, t, :, :, :], input), dim=1)
                        h = getattr(self, name)(xt, h)
                        internal_state[layer_idx] = h
                        xt = h
                    else:
                        h = internal_state[layer_idx]
                        h = getattr(self, name)(xt, h)
                        internal_state[layer_idx] = h
                        xt = h
                else:
                    if layer_idx==0:
                        h = internal_state[layer_idx]
                        mask = y_obs == y_obs
                        x1 = xt
                        x1[mask] = y_obs[mask]
                        input = x1
                        xt = torch.cat((inputs[:, t, :, :, :], input), dim=1)
                        h = getattr(self, name)(xt, h)
                        internal_state[layer_idx] = h
                        xt = h
                    else:
                        h = internal_state[layer_idx]
                        h = getattr(self, name)(xt, h)
                        internal_state[layer_idx] = h
                        xt = h

            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)
        cur_layer_input = layer_output

        layer_output_list.append(layer_output)
        last_state_list.append(h)
        last_state = last_state_list[-1]

        return last_state


if __name__ == "__main__":

    model = DI_ConvGRU(input_size=(20, 24),
                    input_dim=8,
                    hidden_dim=[8, 8, 1],
                    kernel_size=(3,3),
                    num_layers=3).cuda()
    print(model)
    input1 = torch.ones(6, 5, 7, 20, 24).cuda()  # 4(b,t,c,h,w)
    input2 = torch.zeros(6, 5, 1, 20, 24).cuda()
    out = model(input1, input2)
    print(out.shape)

