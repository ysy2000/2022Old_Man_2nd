import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention_Base import Attention
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, encoder_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', 
                 bidirectional_encoder=False, bidirectional_decoder=False,
                 dropout_p=0, use_attention=True):
        super(DecoderRNN, self).__init__()
        
        self.output_size = vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_output_size = encoder_size #* 2 if self.bidirectional_encoder else encoder_size >> Encoder 앞단에서 이미 차원 수 줄임
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.init_input = None
        # bi direcitional RNN과 같이 연산하려면! self.encoder_output_size
        self.rnn = self.rnn_cell(self.hidden_size, self.encoder_output_size, self.n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional_decoder)
        #self.rnn = self.rnn_cell(self.hidden_size, self.hidden_size, self.n_layers,
        #                         batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional_decoder)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout_p)
            
        self.attention = Attention(self.encoder_output_size)

        self.fc = nn.Linear(self.encoder_output_size, self.vocab_size)

        """
        Encoder는 bi-direction 인데 Decoder는 bi가 아니라서 matmul에서의 차이를 해소 하기 위해 
        Decoder output을 그냥 self.encoder_output_size에 맞춰서 늘려버림.
        중간에 지나는거 수정하고 출력때 self.vocab_size 에 맞추도록 조정 ㅠㅠ이게 되나...
        
        """




    def forward_step(self, input_var, hidden, encoder_outputs, context, attn_w, function):
        batch_size = input_var.size(0) # 100
        dec_len = input_var.size(1) # 16
        enc_len = encoder_outputs.size(1)
        enc_dim = encoder_outputs.size(2)

        embedded = self.embedding(input_var) # (B, dec_T, voc_D) -> (B, dec_T, dec_D)
        #print("input_var.size()",input_var.size()) # torch.Size([16, 100])
        #print("embedded.size()",embedded.size())  # torch.Size([16, 100, 512]) 
        embedded = self.input_dropout(embedded)

        y_all = []
        attn_w_all = []

        ## RNN (LSTM/GRU)
        #if self.training:
        #    self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)
        #print("output after RNN",output.size()) # torch.Size([16, 100, 512]) >> 
        # output : rnn(lstm/GRU)의 결과
        # hidden : t에서 게이트를 거친 ht 출력

        ##Attention
        attn = None
        if self.use_attention:
            #attention의 순전파
            output, attn = self.attention(output, encoder_outputs)
            #print("output.size()",output.size()) #torch.Size([16, 100, 1024])

        fc_out = self.fc(output.contiguous().view(-1, self.encoder_output_size)) #torch.Size([3200, 1219])
        #print("fc_out.size()",fc_out.size())
        function_out = function(fc_out, dim=1) #torch.Size([3200, 1219])
        #print("function_out.size()",function_out.size()) 
        predicted_softmax = function_out.view(batch_size, dec_len, -1) 
        #predicted_softmax = function_out.view(batch_size, dec_len, self.output_size )

        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        """
        param:inputs: Decoder inputs sequence, Shape=(B, dec_T)
        param:encoder_hidden: Encoder last hidden states, Default : None
        param:encoder_outputs: Encoder outputs, Shape=(B,enc_T,enc_D)
        """

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        
        inputs, batch_size, max_length = self._validate_args1(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)        
        """
        if teacher_forcing_ratio != 0:
            inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                function, teacher_forcing_ratio)
        else:
            # TF가 없는 경우
            batch_size = encoder_outputs.size(0)
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            print(">> inputs.size()",inputs.size()) #[256, 1]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        """

        decoder_hidden = None
        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2)) # (B, D)
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1)) # (B, T)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn_w = self.forward_step(decoder_input, 
                                                                                decoder_hidden, 
                                                                                encoder_outputs,
                                                                                context,    
                                                                                attn_w, 
                                                                                function=function)
            #print("decoder_output.size()",decoder_output.size()) #forward_step
            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                decode(di, step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, attn_w = self.forward_step(decoder_input, 
                                                                                    decoder_hidden,
                                                                                    encoder_outputs,
                                                                                    context,
                                                                                    attn_w,
                                                                                    function=function)
                #print("decoder_output.size()",decoder_output.size())
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output)
                decoder_input = symbols
        #print("len(decoder_outputs)",len(decoder_outputs))
        #print("len(decoder_outputs[0])",len(decoder_outputs[0]))
        #print("len(decoder_outputs[0][0])",len(decoder_outputs[0][0]))
        return decoder_outputs # 100, 16, 2438 -> 1219

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)
        #print("input is?",inputs)
        if inputs is None:
            #print(" batch_size", batch_size)
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            #print(">> inputs",inputs)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

    def _validate_args1(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
