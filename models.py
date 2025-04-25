import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math

class RNN(nn.Module):
    def __init__(self, input_sz, hidden_sz, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(RNN, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.device = device
        self.W_h = nn.Linear(input_sz + hidden_sz, hidden_sz)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(self.device),
                    torch.zeros(batch_size, self.hidden_size).to(self.device))

        hidden_states = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            common_input = torch.cat([x_t, h_t], dim=-1)
            h_t = torch.tanh(self.W_h(common_input))
            hidden_states.append(h_t.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states
    
class LSTM(nn.Module):
    
    def __init__(self, input_sz, hidden_sz, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        super(LSTM, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.device = device
        #c_t: new memory (the new memory)
        self.W_c = nn.Linear(input_sz + hidden_sz, hidden_sz)

        #i_t: input gate (how much to take from the new memory)
        self.W_i = nn.Linear(input_sz + hidden_sz, hidden_sz)

        #f_t: forget gate (how much to forget from the old memory)
        self.W_f = nn.Linear(input_sz + hidden_sz, hidden_sz)

        #o_t: output gate (how much to take from the new memory to represent the output)
        self.W_o = nn.Linear(input_sz + hidden_sz, hidden_sz)

        self.init_weights()
        
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()  
        
        # initialize h_t and c_t to zeros
        h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(self.device), 
                    torch.zeros(batch_size,self.hidden_size).to(self.device))
        
        hidden_states = []
        
        for t in range(seq_len):
            # get the input at the current timestep 
            x_t = x[:, t, :]
            
            # run the LSTM Cell
            common_input = torch.cat([x_t, h_t], dim = -1)
            
            new_c = torch.tanh(self.W_c(common_input))
            i_t = torch.sigmoid(self.W_i(common_input))
            f_t = torch.sigmoid(self.W_f(common_input))
            c_t = f_t * c_t + i_t * new_c
            
            o_t = torch.sigmoid(self.W_o(common_input))
            h_t = o_t * torch.tanh(c_t)
            
            # save the hidden states in a list
            hidden_states.append(h_t.unsqueeze(1))
            
        hidden_states = torch.cat(hidden_states, dim = 1)
            
        return hidden_states
    
class BasicNet(nn.Module):
    """
    The embeddings are learned from scratch, this model is used for utterance-level classification.
    The inputs are assumed to be padded/truncated to a certain max_sequence_length. 
    For example we feed in a single utterance of length 88 padded to 100. The context 
    in this case are just the words in the utterance.
    """

    def __init__(self, input_embedding_size, hidden_size, 
                 num_words, num_classes, pooling_type = 'last_hidden_state'):
        
        super().__init__()

        self.embedding = nn.Embedding(num_words, input_embedding_size)
        # self.rnn = RNN(input_embedding_size, hidden_size) # Uncomment for RNN (and change in forward pass)
        self.lstm = LSTM(input_embedding_size, hidden_size)
        self.classifier = nn.Linear(hidden_size + 2, num_classes)
        self.pooling_type = pooling_type
        
    def forward(self, x, speaker_ids):
        
        # x is of shape (batch_size, max_seq_length)
        x = self.embedding(x)   # (batch_size, seq_len, embed_dim)
        x = self.lstm(x)        # (batch_size, seq_len, hidden_size)

        if self.pooling_type == "last_hidden_state":
            x = x[:, -1, :]     # (batch_size, hidden_size)
        else:
            x = x.mean(1)       # (batch_size, hidden_size)

        x = torch.cat((speaker_ids, x), dim=1)  # (batch_size, hidden_size + 2)

        x = self.classifier(x)         # (batch_size, len(acts_labels))
        return x
    
    def extract_utterance_vector(self, x, speaker_ids):
        x = self.embedding(x)
        x = self.lstm(x)
        if self.pooling_type == "last_hidden_state":
            return x[:, -1, :]  # (batch_size, hidden_size)
        else:
            return x.mean(1)    # (batch_size, hidden_size)

    
class ContextNet(nn.Module):
    """
    The embeddings are assumed to be pre-trained. This model is used for classifying based on a context of utterances.
    In this case each utterance is pre-computed into an embedding of same size. 
    The context is the sequence of utterances (in the form of rich embeddings), 
    and we no longer consider single words of an utterance.
    """

    def __init__(self, input_size, hidden_size, num_classes, pooling_type='last_hidden_state'):
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size)  # Use LSTM by default
        self.classifier = nn.Linear(hidden_size, num_classes)  # Output layer
        self.pooling_type = pooling_type

    def forward(self, x):
        # x is of shape (batch_size, seq_len, input_size)
        x = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        # Pooling
        if self.pooling_type == "last_hidden_state":
            x = x[:, -1, :]  # (batch_size, hidden_size)
        else:
            x = x.mean(dim=1)  # (batch_size, hidden_size)

        # Classify
        x = self.classifier(x)  # (batch_size, num_classes)
        return x
    
    def extract_utterance_vector(self, x):
        x = self.lstm(x)
        if self.pooling_type == "last_hidden_state":
            return x[:, -1, :]  # (batch_size, hidden_size)
        else:
            return x.mean(dim=1)    # (batch_size, hidden_size)

    
# ---------------------- Models from pytorch fork ----------------------
# (https://github.com/guillitte/pytorch-sentiment-neuron) needed for loading pretrained model

class mLSTM(nn.Module):

	def __init__(self, data_size, hidden_size, n_layers = 1):
		super(mLSTM, self).__init__()
        
		self.hidden_size = hidden_size
		self.data_size = data_size
		self.n_layers = n_layers
		input_size = data_size + hidden_size
        
       
		self.wx = nn.Linear(data_size, 4*hidden_size, bias = False)
		self.wh = nn.Linear(hidden_size, 4*hidden_size, bias = True)
		self.wmx = nn.Linear(data_size, hidden_size, bias = False) 
		self.wmh = nn.Linear(hidden_size, hidden_size, bias = False) 
  
	def forward(self, data, last_hidden):

		hx, cx = last_hidden
		m = self.wmx(data) * self.wmh(hx)
		gates = self.wx(data) + self.wh(m)
		i, f, o, u = gates.chunk(4, 1)

		i = F.sigmoid(i)
		f = F.sigmoid(f)
		u = F.tanh(u)
		o = F.sigmoid(o)

		cy = f * cx + i * u
		hy = o * F.tanh(cy)

		return hy, cy

class StackedLSTM(nn.Module):
    def __init__(self, cell, num_layers, input_size, rnn_size, output_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.h2o = nn.Linear(rnn_size, output_size)
        
        self.layers = []
        for i in range(num_layers):
            layer = cell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            if i == 0:
                input = h_1_i
            else:
                input = input + h_1_i
            if i != len(self.layers):
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        output = self.h2o(input)

        return (h_1, c_1),output
        
    def state0(self, batch_size):
            h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size), requires_grad=False)
            c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size), requires_grad=False)
            return (h_0, c_0) 
