from torch import nn
import torch



'''
Takes the n=Batch matrixes (Seq_len, d_model) to output the probabilities of each word(or item) in the vocab
'''

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int)-> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, Vocab_size)
        return torch.log_softmax(self.proj[x], dim = -1)
    

    

        
