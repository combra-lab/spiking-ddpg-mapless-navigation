import numpy as np
import nxsdk.api.n2a as nx
from nxsdk.net.groups import CompartmentGroup, ConnectionGroup
from nxsdk.graph.monitor.probes import SpikeProbeCondition
from nxsdk.graph.processes.phase_enums import Phase
import os
import sys
sys.path.append('../../')
from evaluation.loihi_network.utility import pytorch_trained_snn_param_2_loihi_snn_param_amp


class SpikingActorNet:
    """ SNN for Goal Reaching Task on Loihi"""
    def __init__(self,
                 weights,
                 bias,
                 core_list,
                 bias_amp=(1, 1, 1, 1),
                 vth=0.5,
                 cdecay=1/2,
                 vdecay=1/4):
        """
        :param weights: raw weights from trained SNN
        :param bias: raw bias from trained SNN
        :param core_list: core list of each layer and bias neurons
        :param bias_amp: use multiple bias neurons if bias is large
        :param vth: raw voltage threshold of trained neuron
        :param cdecay: raw current decay of trained neuron
        :param vdecay: raw voltage decay of trained neuron
        """
        assert isinstance(weights, list) and isinstance(bias, list)
        assert len(weights) == len(bias)
        assert isinstance(core_list, list) and len(core_list) == len(weights) + 2
        self.net = nx.NxNet()
        self.core_list = core_list
        self.bias_amp = bias_amp
        self.loihi_cdecay = int(cdecay * 2**12)
        self.loihi_vdecay = int(vdecay * 2**12)
        self.loihi_weights = {'pos_w': [], 'neg_w': [], 'pos_w_mask': [], 'neg_w_mask': []}
        self.loihi_bias = {'pos_b': [], 'neg_b': [], 'pos_b_mask': [], 'neg_b_mask': []}
        self.loihi_vth = []
        self.loihi_scale_factor = []
        self.loihi_snn_dimension = []
        self.loihi_bias_start_end = [0]
        bias_ita = 0
        for num in range(len(weights)):
            param_dict, new_vth, new_sf = pytorch_trained_snn_param_2_loihi_snn_param_amp(
                weights[num], bias[num], vth, self.bias_amp[num]
            )
            self.loihi_weights['pos_w'].append(param_dict['pos_w'])
            self.loihi_weights['neg_w'].append(param_dict['neg_w'])
            self.loihi_weights['pos_w_mask'].append(param_dict['pos_w_mask'])
            self.loihi_weights['neg_w_mask'].append(param_dict['neg_w_mask'])
            self.loihi_bias['pos_b'].append(param_dict['pos_b'])
            self.loihi_bias['neg_b'].append(param_dict['neg_b'])
            self.loihi_bias['pos_b_mask'].append(param_dict['pos_b_mask'])
            self.loihi_bias['neg_b_mask'].append(param_dict['neg_b_mask'])
            self.loihi_vth.append(new_vth)
            self.loihi_scale_factor.append(new_sf)
            if num == 0:
                self.loihi_snn_dimension.append(param_dict['pos_w'].shape[1])
            self.loihi_snn_dimension.append(param_dict['pos_w'].shape[0])
            bias_ita += bias_amp[num]
            self.loihi_bias_start_end.append(bias_ita)
        # network layers
        self.network_input_layer = None
        self.network_bias_layer = None
        self.network_hidden_layer = None
        self.network_output_layer = None
        # Online input connections
        self.pseudo_2_input = None
        self.pseudo_2_bias = None

    def set_network_input_layer(self):
        """
        Setup Network Input Layer
        """
        assert len(self.core_list[0]) == self.loihi_snn_dimension[0]
        neuron_prototype = nx.CompartmentPrototype(
            compartmentCurrentDecay=2**12
        )
        self.network_input_layer = self.net.createCompartmentGroup(size=0)
        for core in self.core_list[0]:
            input_neuron = self.net.createCompartment(prototype=neuron_prototype)
            input_neuron.logicalCoreId = core
            self.network_input_layer.addCompartments(input_neuron)

    def set_online_input_encoding(self):
        """
        Create Fake Connections for online input encoding
        :return online_fanin_axon_id
        """
        assert isinstance(self.network_input_layer, CompartmentGroup)
        assert len(self.core_list[0]) == self.loihi_snn_dimension[0]
        neuron_prototype = nx.CompartmentPrototype()
        pseudo_neurons = self.net.createCompartmentGroup(size=0)
        for core in self.core_list[0]:
            pseudo_single = self.net.createCompartment(prototype=neuron_prototype)
            pseudo_single.logicalCoreId = core
            pseudo_neurons.addCompartments(pseudo_single)
        conn_w = np.eye(self.loihi_snn_dimension[0]) * 120
        conn_mask = np.int_(np.eye(self.loihi_snn_dimension[0]))
        self.pseudo_2_input = pseudo_neurons.connect(
            self.network_input_layer,
            prototype=nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY),
            connectionMask=conn_mask,
            weight=conn_w
        )

    def get_online_input_axon_id(self):
        """
        Get axon id for input layer
        :return:
        """
        assert isinstance(self.pseudo_2_input, ConnectionGroup)
        online_fanin_axon_id = []
        for conn in self.pseudo_2_input:
            online_fanin_axon_id.append(self.net.resourceMap.inputAxon(conn.inputAxon.nodeId))
        return online_fanin_axon_id

    def set_network_bias_layer(self):
        """
        Setup Network Bias Layer
        """
        assert len(self.core_list[-1]) == self.loihi_bias_start_end[-1]
        neuron_prototype = nx.CompartmentPrototype(
            compartmentCurrentDecay=2**12
        )
        self.network_bias_layer = self.net.createCompartmentGroup(size=0)
        for core in self.core_list[-1]:
            bias_neuron = self.net.createCompartment(prototype=neuron_prototype)
            bias_neuron.logicalCoreId = core
            self.network_bias_layer.addCompartments(bias_neuron)

    def set_online_bias_encoding(self):
        """
        Create Fake Connections for online bias encoding
        :return online_fanin_axon_id
        """
        assert isinstance(self.network_bias_layer, CompartmentGroup)
        assert len(self.core_list[-1]) == self.loihi_bias_start_end[-1]
        neuron_prototype = nx.CompartmentPrototype()
        pseudo_neurons = self.net.createCompartmentGroup(size=0)
        for core in self.core_list[-1]:
            pseudo_single = self.net.createCompartment(prototype=neuron_prototype)
            pseudo_single.logicalCoreId = core
            pseudo_neurons.addCompartments(pseudo_single)
        conn_w = np.eye(self.loihi_bias_start_end[-1]) * 120
        conn_mask = np.int_(np.eye(self.loihi_bias_start_end[-1]))
        self.pseudo_2_bias = pseudo_neurons.connect(
            self.network_bias_layer,
            prototype=nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY),
            connectionMask=conn_mask,
            weight=conn_w
        )

    def get_online_bias_axon_id(self):
        """
        Get axon id for bias layer
        :return:
        """
        assert isinstance(self.pseudo_2_bias, ConnectionGroup)
        online_fanin_axon_id = []
        for conn in self.pseudo_2_bias:
            online_fanin_axon_id.append(self.net.resourceMap.inputAxon(conn.inputAxon.nodeId))
        return online_fanin_axon_id

    def set_network_hidden_layer(self):
        """
        Setup Network Hidden Layer
        """
        for l in range(len(self.loihi_snn_dimension) - 2):
            assert len(self.core_list[l+1]) == self.loihi_snn_dimension[l+1]
        self.network_hidden_layer = []
        for layer in range(len(self.loihi_snn_dimension) - 2):
            neuron_prototype = nx.CompartmentPrototype(
                vThMant=self.loihi_vth[layer],
                compartmentCurrentDecay=self.loihi_cdecay,
                compartmentVoltageDecay=self.loihi_vdecay
            )
            current_layer = self.net.createCompartmentGroup(size=0)
            for core in self.core_list[layer+1]:
                single_neuron = self.net.createCompartment(prototype=neuron_prototype)
                single_neuron.logicalCoreId = core
                current_layer.addCompartments(single_neuron)
            self.network_hidden_layer.append(current_layer)

    def set_network_output_layer(self):
        """
        Setup Network Output Layer
        """
        assert len(self.core_list[-2]) == self.loihi_snn_dimension[-1]
        neuron_prototype = nx.CompartmentPrototype(
            vThMant=self.loihi_vth[-1],
            compartmentCurrentDecay=self.loihi_cdecay,
            compartmentVoltageDecay=self.loihi_vdecay
        )
        self.network_output_layer = self.net.createCompartmentGroup(size=0)
        for core in self.core_list[-2]:
            single_neuron = self.net.createCompartment(prototype=neuron_prototype)
            single_neuron.logicalCoreId = core
            self.network_output_layer.addCompartments(single_neuron)

    def set_online_output_decoding(self):
        """
        Create Fake Spike Probes for online output layer decoding
        """
        assert isinstance(self.network_output_layer, CompartmentGroup)
        custom_probe_cond = SpikeProbeCondition(tStart=1000000000000)
        pseudo_spike_probe = self.network_output_layer.probe(nx.ProbeParameter.SPIKE, custom_probe_cond)

    def set_network_connections(self):
        """
        Setup Connections in the network
        """
        assert isinstance(self.network_input_layer, CompartmentGroup)
        assert isinstance(self.network_bias_layer, CompartmentGroup)
        assert isinstance(self.network_output_layer, CompartmentGroup)
        assert isinstance(self.network_hidden_layer, list)
        for l in range(len(self.loihi_snn_dimension) - 2):
            assert isinstance(self.network_hidden_layer[l], CompartmentGroup)
        for layer in range(len(self.loihi_snn_dimension) - 1):
            if layer == 0:
                pre_neuron = self.network_input_layer
                post_neuron = self.network_hidden_layer[layer]
            elif layer == len(self.loihi_snn_dimension) - 2:
                pre_neuron = self.network_hidden_layer[layer-1]
                post_neuron = self.network_output_layer
            else:
                pre_neuron = self.network_hidden_layer[layer-1]
                post_neuron = self.network_hidden_layer[layer]
            pre_neuron.connect(
                post_neuron,
                prototype=nx.ConnectionPrototype(
                    numWeightBits=8,
                    signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY
                ),
                connectionMask=self.loihi_weights['pos_w_mask'][layer],
                weight=self.loihi_weights['pos_w'][layer]
            )
            pre_neuron.connect(
                post_neuron,
                prototype=nx.ConnectionPrototype(
                    numWeightBits=8,
                    signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY
                ),
                connectionMask=self.loihi_weights['neg_w_mask'][layer],
                weight=self.loihi_weights['neg_w'][layer]
            )
            bias_pos_conn_w = np.zeros((self.loihi_snn_dimension[layer+1],
                                        self.loihi_bias_start_end[-1]))
            bias_pos_conn_mask = np.int_(np.zeros((self.loihi_snn_dimension[layer+1],
                                                   self.loihi_bias_start_end[-1])))
            bias_pos_conn_w[:, self.loihi_bias_start_end[layer]:self.loihi_bias_start_end[layer+1]] = self.loihi_bias['pos_b'][layer][:, :]
            bias_pos_conn_mask[:, self.loihi_bias_start_end[layer]:self.loihi_bias_start_end[layer+1]] = self.loihi_bias['pos_b_mask'][layer][:, :]
            self.network_bias_layer.connect(
                post_neuron,
                prototype=nx.ConnectionPrototype(
                    numWeightBits=8,
                    signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY
                ),
                connectionMask=bias_pos_conn_mask,
                weight=bias_pos_conn_w
            )
            bias_neg_conn_w = np.zeros((self.loihi_snn_dimension[layer+1],
                                        self.loihi_bias_start_end[-1]))
            bias_neg_conn_mask = np.int_(np.zeros((self.loihi_snn_dimension[layer+1],
                                                   self.loihi_bias_start_end[-1])))
            bias_neg_conn_w[:, self.loihi_bias_start_end[layer]:self.loihi_bias_start_end[layer+1]] = self.loihi_bias['neg_b'][layer][:, :]
            bias_neg_conn_mask[:, self.loihi_bias_start_end[layer]:self.loihi_bias_start_end[layer+1]] = self.loihi_bias['neg_b_mask'][layer][:, :]
            self.network_bias_layer.connect(
                post_neuron,
                prototype=nx.ConnectionPrototype(
                    numWeightBits=8,
                    signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY
                ),
                connectionMask=bias_neg_conn_mask,
                weight=bias_neg_conn_w
            )

    def setup_snn(self,
                  print_axon=False,
                  snip_dir='../loihi_network/snip',
                  encode_input_num=6,
                  decode_output_num=2):
        """
        Setup SNN on Loihi
        """
        self.set_network_output_layer()
        self.set_online_output_decoding()
        self.set_network_input_layer()
        self.set_online_input_encoding()
        self.set_network_bias_layer()
        self.set_online_bias_encoding()
        self.set_network_hidden_layer()
        self.set_network_connections()
        compiler = nx.N2Compiler()
        board = compiler.compile(self.net)
        if print_axon:
            input_axon_id = self.get_online_input_axon_id()
            bias_axon_id = self.get_online_bias_axon_id()
            print("Axon Id For Input Layer: ")
            print(input_axon_id)
            print("Axon Id For Bias Layer: ")
            print(bias_axon_id)
        include_dir = os.path.abspath(snip_dir)
        encoder_snip = board.createSnip(
            Phase.EMBEDDED_SPIKING,
            name='encoder',
            includeDir=include_dir,
            cFilePath=include_dir + '/encoder.c',
            funcName='run_encoder',
            guardName='do_encoder'
        )
        decoder_snip = board.createSnip(
            Phase.EMBEDDED_MGMT,
            name='decoder',
            includeDir=include_dir,
            cFilePath=include_dir + '/decoder.c',
            funcName='run_decoder',
            guardName='do_decoder'
        )
        encoder_channel = board.createChannel(b'encodeinput', "int", encode_input_num)
        encoder_channel.connect(None, encoder_snip)
        decoder_channel = board.createChannel(b'decodeoutput', "int", decode_output_num)
        decoder_channel.connect(decoder_snip, None)
        return board, encoder_channel, decoder_channel

