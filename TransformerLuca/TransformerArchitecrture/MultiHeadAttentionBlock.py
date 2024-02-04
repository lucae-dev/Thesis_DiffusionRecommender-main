import math
import torch 
import torch.nn as nn 

class MultiHeadAttentionBlock(nn.module):

    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model% h == 0, "d_model is not divisable by h(MultiHeadAttentionBlock)"#checking d_model is divisable by number of heads
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) #query layer
        self.w_k = nn.Linear(d_model,d_model) #key layer
        self.w_v = nn.Linear(d_model,d_model) # value layer

        self.w_o = nn.Linear(d_model,d_model) #output layer
        self.dropout = dropout

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, Seq_Len,d_k) @((Batch, h, d_k, Seq_Len,))--> (Batch, h, Seq_Len, Seq_Len )
        attention_scores = (query @ key.transpose(-2,-1)/math.sqrt(d_k))
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) 
        if dropout is not None:
           attention_scores = dropout(attention_scores)

        return (attention_scores@value), attention_scores

        #mask avoids some input to see at others (for sequence models is useful to not make previous words the future ones)
    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k) #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v) #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        #here we basically divide the 'embeddings' of dimension domodel in h parts of dimension d_k
        #(Batch, Seq_Len, d_model)-- >(Batch,Seq_Len, h, d_k) --transpose-->(Batch,h,Seq_Len,d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #transpose switches the second(1) and third (2)dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #(Batch, h, Seq_Len, d_k) -- > (Batch, Seq_Len, h, d_k) --> (Batch,Seq_Len, d_model)
        x = x.transpose(1,2).contigous().view(x.shape[0], -1, self.h * self.d_k)


        return self.w_o(x)