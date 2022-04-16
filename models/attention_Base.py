"""

Copyright 2017- IBM Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    #Decoder에서 받은 hidden_size가 dim
    def __init__(self, dim):
        super(Attention, self).__init__()   #Module(Attention,self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    # dim????????????????????
    def forward(self, output, context):
        # output    : rnn의 output(t에서 게이트를 거친 ht)
        #  (batch, out_len, dim)
        # context   : 인코더의 output - hs
        #  (batch, in_len, dim)
        batch_size = output.size(0)
        hidden_size = output.size(2) #dim = hidden size = 글자(단어)를 표현하는 고정길이 벡터
        input_size = context.size(1)

        ### hs dot ht = s
        #print("output",output.size()) #torch.Size([16, 100, 512])
        #print("context",context.size()) #torch.Size([16, 717, 1024]) 
        #print("context.transpose(1, 2)",context.transpose(1, 2).size()) #torch.Size([16, 1024, 717])
        attn = torch.bmm(output, context.transpose(1, 2))
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        ### s -> a  : (batch, out_len, in_len)
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        ###
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # 기존 Attention에 추가적인 작업? ht와 context vector를 합친후 FC
        # concat -> (batch, out_len, 2*dim)
        # mix : (batch, out_len, dim)
        # out : (batch, out_len, dim)
        #print("output.size()",output.size())#torch.Size([16, 100, 1024])
        #print("mix.size()",mix.size())    #torch.Size([16, 100, 1024])    
        combined = torch.cat((mix, output), dim=2)
        #print("combined.size()",combined.size()) #torch.Size([16, 100, 2048])  
        #print("combined.view(-1, 2 * hidden_size).size()",combined.view(-1, 2 * hidden_size).size())  #torch.Size([1600, 2048])  

        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
