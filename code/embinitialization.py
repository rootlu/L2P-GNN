import torch


class EmbInitial(torch.nn.Module):
    def __init__(self, emb_dim, node_in_channels):
        super(EmbInitial, self).__init__()

        self.input_node_embeddings = torch.nn.Embedding(node_in_channels+1, emb_dim)
        torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

    def forward(self, node_fea):
        return self.input_node_embeddings(node_fea.to(torch.int64).view(-1, ))


class EmbInitial_DBLP(torch.nn.Module):
    def __init__(self, emb_dim, node_in_channels):
        super(EmbInitial_DBLP, self).__init__()

        self.input_node_embeddings = torch.nn.Linear(node_in_channels, emb_dim)
        torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

    def forward(self, node_fea):
        return self.input_node_embeddings(node_fea)


class EmbInitial_CHEM(torch.nn.Module):
    def __init__(self, emb_dim, node_in_channels):
        super(EmbInitial_CHEM, self).__init__()

        num_atom_type = 120
        num_chirality_tag = 3

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

    def forward(self, node_fea):
        return self.x_embedding1(node_fea[:,0]) + self.x_embedding2(node_fea[:,1])


