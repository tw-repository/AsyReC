"""
Overall framework
"""

from transformers import AutoTokenizer, AutoModel
from torchvision.models.video import r3d_18
import os
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

warnings.filterwarnings("ignore", category=UserWarning, message=".*no max_length.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE = 768
dim = 768
heads = 8
dim_head = 64
dropout = 0.
depth = 2
mlp_dim = 1024

# load pre-trained AST model
AST_path = '/your_path/pretrianedModels/ast-finetuned-audioset-10-10-0.4593'
AST_model = ASTForAudioClassification.from_pretrained(AST_path).to(device)
AST_model.eval()
print("-----------------> pretrained Audio Spectrogram Transformer (AST) model load successfully! <-----------------")


# load pre-trained BERT model
Bert_path = '/your_path/pretrianedModels/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(Bert_path)
lan_model = AutoModel.from_pretrained(Bert_path).to(device)
lan_model.eval()
print("-----------------> pretrained bert model load successfully! <-----------------")


class GetSentenceFeatures(nn.Module):
    def __init__(self, tokenizer, lan_model):
        super(GetSentenceFeatures, self).__init__()
        self.tokenizer = tokenizer
        self.lan_model = lan_model

    def forward(self, text_list):
        sentence_features = []
        for text in text_list:
            lan_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            lan_token = {k: v.to(self.lan_model.device) for k, v in lan_token.items()}

            with torch.no_grad():
                outputs = self.lan_model(**lan_token)
            last_hidden_states = outputs.last_hidden_state
            cls_index = lan_token['input_ids'].squeeze().tolist().index(self.tokenizer.cls_token_id)
            sentence_feature = last_hidden_states[:, cls_index, :].detach()
            sentence_features.append(sentence_feature)

        sentence_tensor = torch.stack(sentence_features, 0).squeeze(dim=1)
        if sentence_tensor.dim() == 1:
            sentence_tensor = sentence_tensor.unsqueeze(0)
        return sentence_tensor


class GetAudioFeatures(nn.Module):
    def __init__(self):
        super(GetAudioFeatures, self).__init__()

    def forward(self, audio_tensor):
        audio_feature = []
        num_chunks = audio_tensor.size(0)
        audio_list = torch.chunk(audio_tensor, chunks=num_chunks, dim=0)
        for audio in audio_list:
            with torch.no_grad():
                outputs = AST_model(audio.squeeze(dim=0).to(AST_model.device), output_hidden_states=True)
            audio_feature.append(outputs.hidden_states[-1])

        audio_tensor = torch.stack(audio_feature, 0).squeeze(dim=1)
        audio_feature_pooled = torch.mean(audio_tensor, dim=1).squeeze(0)
        if audio_feature_pooled.dim() == 1:
            audio_feature_pooled = audio_feature_pooled.unsqueeze(0)
        return audio_feature_pooled


class person_pair(nn.Module):

    def __init__(self):
        super(person_pair, self).__init__()

        self.face_a = r3d_18(pretrained=True)
        self.face_b = self.face_a
        self.person_a = r3d_18(pretrained=True)
        self.person_b = self.person_a

        for model in [self.face_a, self.face_b, self.person_a, self.person_b]:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, SIZE, bias=True)

    def forward(self, face_A, person_A, face_B, person_B):
        face_a = self.face_a(face_A)
        face_b = self.face_b(person_A)
        person_a = self.person_a(face_B)
        person_b = self.person_b(person_B)

        return face_a, face_b, person_a, person_b


class adjGenerator(nn.Module):
    def __init__(self, num_modalities):
        super(adjGenerator, self).__init__()
        self.num_modalities = num_modalities

    def forward(self, features):
        batch_size = features.size(0)
        adj = torch.ones(batch_size, self.num_modalities, self.num_modalities).to(features.device)
        mask = torch.eye(self.num_modalities).bool().unsqueeze(0).repeat(batch_size, 1, 1).to(features.device)
        adj[mask] = 0

        return adj


class adjInterGenerator(nn.Module):
    def __init__(self, num_modalities):
        super(adjInterGenerator, self).__init__()
        self.num_modalities = num_modalities

    def forward(self, features):
        batch_size = features.size(0)

        total_nodes = self.num_modalities * 2
        adj_interaction = torch.zeros(batch_size, total_nodes, total_nodes)

        for i in range(self.num_modalities):
            for j in range(self.num_modalities):
                adj_interaction[:, i, j + self.num_modalities] = 1
                adj_interaction[:, j + self.num_modalities, i] = 1

        return adj_interaction.to(features.device)


class NodeAttention(nn.Module):
    def __init__(self, in_features, dropout=0.6, alpha=0.2):
        super(NodeAttention, self).__init__()
        self.a = nn.Parameter(torch.empty(size=(in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):

        batch_size, num_modalties, _ = h.size()
        e = torch.bmm(h, self.a.unsqueeze(0).expand(batch_size, -1, -1))
        attention = F.softmax(self.leakyrelu(e), dim=1)
        attention = self.dropout(attention)
        weighted_h = h * attention

        return weighted_h + h


class NE_AGN(nn.Module):
    def __init__(self, in_features, out_features, num_modalties, heads=1, multi_head=False, dropout=0.6, alpha=0.2,
                 concat=True):
        super(NE_GAT, self).__init__()
        assert out_features % heads == 0, "Out features must be divisible by number of heads"

        self.multi_head = multi_head
        self.heads = heads
        self.in_features = in_features
        self.out_features_per_head = out_features // heads

        if multi_head:
            self.W = nn.Parameter(torch.empty(size=(heads, in_features, self.out_features_per_head)))
        else:
            self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)

        if multi_head:
            self.a = nn.Parameter(torch.empty(size=(heads, 2 * self.out_features_per_head, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.node_attention = NodeAttention(in_features, dropout, alpha)
        self.num_modalties = num_modalties
        self.out_features = out_features

    def forward(self, h, adj, node_att, edge_att):

        batch_size, num_modalties, _ = h.size()
        if node_att is None:
            weighted_h = h
        else:
            weighted_h = self.node_attention(h)
        if edge_att is None:
            Wh = torch.bmm(weighted_h, self.W1.unsqueeze(0).expand(batch_size, -1, -1))
            h_prime = torch.bmm(adj, Wh)
        else:
            if self.multi_head:
                weighted_h_expanded = weighted_h.unsqueeze(2).expand(
                    -1, -1, self.heads, -1).contiguous().view(batch_size, num_modalties, self.heads, self.in_features)
                W_expanded = self.W.unsqueeze(0).expand(batch_size, -1, -1, -1)

                Wh = torch.matmul(weighted_h_expanded, W_expanded).squeeze(-2)

                Wh1 = torch.matmul(Wh, self.a[:, :self.out_features_per_head, :].unsqueeze(0).expand(batch_size, -1, -1, -1)).transpose(2, 3)
                Wh2 = torch.matmul(Wh, self.a[:, self.out_features_per_head:, :].unsqueeze(0).expand(batch_size, -1, -1, -1))
                e = Wh1 + Wh2
            else:
                Wh = torch.bmm(weighted_h, self.W.unsqueeze(0).expand(batch_size, -1, -1))
                Wh1 = torch.matmul(Wh, self.a[:self.out_features, :].unsqueeze(0).expand(batch_size, -1, -1)).transpose(1, 2)
                Wh2 = torch.matmul(Wh, self.a[self.out_features:, :].unsqueeze(0).expand(batch_size, -1, -1))
                e = Wh1 + Wh2

            zero_vec = -9e15 * torch.ones_like(e)

            if self.multi_head:
                adj_expanded = adj.unsqueeze(1).expand(-1, self.heads, -1, -1)
                attention = torch.where(adj_expanded > 0, e, zero_vec)
            else:
                attention = torch.where(adj > 0, e, zero_vec)

            attention = F.softmax(attention, dim=-1)
            attention = self.dropout(attention)

            if self.multi_head:
                Wh_reshaped = Wh.view(batch_size, num_modalties, self.heads, self.out_features_per_head).permute(0, 2, 1, 3)
                h_prime = torch.matmul(attention, Wh_reshaped)
                h_prime = h_prime.contiguous().view(batch_size, num_modalties, self.heads * self.out_features_per_head)
            else:
                h_prime = torch.bmm(attention, Wh)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime


def normalize(tensor, max_values):
    max_values = torch.clamp(max_values, min=1e-8)

    even_mask = tensor % 2 == 0

    relative_position = torch.zeros_like(tensor, dtype=torch.float32)
    relative_position[even_mask] = torch.sin(tensor[even_mask] / max_values[even_mask] * math.pi)
    relative_position[~even_mask] = torch.cos(tensor[~even_mask] / max_values[~even_mask] * math.pi)

    return relative_position


class network(nn.Module):
    def __init__(self, num_classes, num_modalities=4):
        super(network, self).__init__()
        self.num_modalities = num_modalities
        self.person_pair = person_pair()
        self.text_feature = GetSentenceFeatures(tokenizer, lan_model)
        self.audio_feature = GetAudioFeatures()

        self.adjInner = adjGenerator(num_modalities)
        self.adjInter = adjInterGenerator(num_modalities)

        self.graphInference1 = NE_AGN(SIZE, SIZE, num_modalities, heads=num_modalities, multi_head=True)
        self.graphInference2 = NE_AGN(SIZE, SIZE, num_modalities, heads=num_modalities, multi_head=True)
        self.graphInference3 = NE_AGN(SIZE, SIZE, num_modalities * 2, heads=num_modalities * 2, multi_head=True)

        self.fc_upSample = nn.Linear(1, SIZE)

        self.fc_fusion = nn.Linear(SIZE * 4, SIZE)
        self.fc_fusionV = nn.Linear(SIZE * 4, SIZE)
        self.fc_fusionA = nn.Linear(SIZE * 2, SIZE)
        self.fc_fusionT = nn.Linear(SIZE * 2, SIZE)
        self.fc_class = nn.Linear(SIZE * 2, num_classes)
        self.ReLU = nn.ReLU(True)

    def forward(self, face_A, person_A, audio_A, text_A, face_B, person_B, audio_B, text_B, position, video_length):

        face_a, face_b, person_a, person_b = self.person_pair(face_A, person_A, face_B, person_B)
        text_A_feature = self.text_feature(text_A)
        text_B_feature = self.text_feature(text_B)
        audio_A_feature = self.audio_feature(audio_A)
        audio_B_feature = self.audio_feature(audio_B)

        graph_A = torch.stack([face_a, person_a, audio_A_feature, text_A_feature], dim=0)
        graph_B = torch.stack([face_b, person_b, audio_B_feature, text_B_feature], dim=0)

        adj_inner = self.adjInner(face_a)
        adj_inter = self.adjInter(face_a)

        graphs_A = self.graphInference1(graph_A.transpose(0, 1), adj_inner, node_att=True, edge_att=None)
        graphs_B = self.graphInference2(graph_B.transpose(0, 1), adj_inner, node_att=True, edge_att=None)

        graph_inter = torch.cat((graphs_A, graphs_B), dim=1)
        graphs_all = self.graphInference3(graph_inter, adj_inter, node_att=None, edge_att=True)

        normalized_position = normalize(position, video_length)
        upSampled_position = self.fc_upSample(normalized_position.unsqueeze(1))

        fusionV = self.fc_fusionV(torch.cat((face_a, person_a, face_b, person_b), dim=1))
        fusionA = self.fc_fusionA(torch.cat((audio_A_feature, audio_B_feature), dim=1))
        fusionT = self.fc_fusionT(torch.cat((text_A_feature, text_B_feature), dim=1))

        fusionAll = self.fc_fusion(torch.cat([torch.mean(graphs_all, dim=1), fusionV, fusionA, fusionT], dim=1))
        class_all = self.fc_class(self.ReLU(torch.cat([fusionAll, upSampled_position], dim=1)))

        return class_all
