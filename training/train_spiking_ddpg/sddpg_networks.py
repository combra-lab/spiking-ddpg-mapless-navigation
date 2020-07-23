import torch
import torch.nn as nn


"""
Our implementation of STBP on SNN is inspired by 
the open-sourced implementation of STBP for:

Wu, Yujie, Lei Deng, Guoqi Li, Jun Zhu, and Luping Shi. 
"Spatio-temporal backpropagation for training high-performance spiking neural networks." 
Frontiers in neuroscience 12 (2018).

Their implementation for SCNN can be found here:
https://github.com/yjwu17/BP-for-SpikingNN

We would like to thank them for open-source their implementation.
"""


NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class ActorNetSpiking(nn.Module):
    """ Spiking Actor Network """
    def __init__(self, state_num, action_num, device, batch_window=50, hidden1=256, hidden2=256, hidden3=256):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param device: device used
        :param batch_window: window steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNetSpiking, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.pseudo_spike = PseudoSpikeRect.apply
        self.fc1 = nn.Linear(self.state_num, self.hidden1, bias=True)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=True)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3, bias=True)
        self.fc4 = nn.Linear(self.hidden3, self.action_num, bias=True)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, x, batch_size):
        """

        :param x: state batch
        :param batch_size: size of batch
        :return: out
        """
        fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_u = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_v = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_s = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc3_u = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_v = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_s = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc4_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_s = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)
        for step in range(self.batch_window):
            input_spike = x[:, :, step]
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, input_spike, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc3_u, fc3_v, fc3_s = self.neuron_model(self.fc3, fc2_s, fc3_u, fc3_v, fc3_s)
            fc4_u, fc4_v, fc4_s = self.neuron_model(self.fc4, fc3_s, fc4_u, fc4_v, fc4_s)
            fc4_sumspike += fc4_s
        out = fc4_sumspike / self.batch_window
        return out

