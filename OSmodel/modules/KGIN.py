
import math
import random
import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
import os 
from utils.evaluate import sigmoid_function 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class DNN(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, cell_feat_len:int, cell_feat2_len:int, hidden_size: int):
        super(DNN, self).__init__()
        self.dropout=nn.Dropout(p=0.5)

        self.drug3MLP1=DrugMLP(167,drug_feat1_len)
        self.drug3MLP2=DrugMLP(167,drug_feat1_len)
        self.cell3MLP=CellLineMLP(16376,drug_feat1_len)

        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len),
        )

        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, drug_feat2_len),
        )

        # 768 / 2 = 384
        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat_len*2 ),
            nn.Linear(cell_feat_len*2,cell_feat_len ),
        )

        self.cell_network2 = nn.Sequential(
            nn.Linear(cell_feat2_len, cell_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat2_len*2 ),
            nn.Linear(cell_feat2_len*2, cell_feat2_len),
        )

        self.fc_network2 = nn.Sequential(
            nn.BatchNorm1d(9*drug_feat1_len),
            nn.Linear(9*drug_feat1_len,(9*drug_feat1_len)//2 ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d((9*drug_feat1_len)//2),
            nn.Linear((9*drug_feat1_len) //2, (9*drug_feat1_len) //2 // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d((9*drug_feat1_len)  // 2//2),
            nn.Linear((9*drug_feat1_len ) // 2 // 2, 1)
        )
        self.fc_network1 = nn.Sequential(
            nn.BatchNorm1d(3*(drug_feat1_len + drug_feat2_len+cell_feat_len)),
            nn.Linear(3*(drug_feat1_len + drug_feat2_len+cell_feat_len), hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

        
        

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug1_3: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor,drug2_3: torch.Tensor, cell_feat1: torch.Tensor, cell_feat2: torch.Tensor,cell_3: torch.Tensor):
        #drug1_feat1_vector = self.drug_network1( drug1_feat1 ) 
        #drug1_feat2_vector = self.drug_network2( drug1_feat2 )
        #drug2_feat1_vector = self.drug_network1( drug2_feat1 ) 
        #drug2_feat2_vector = self.drug_network2( drug2_feat2 )
        #cell_feat_vector = self.cell_network(cell_feat)
        #cell_feat_vector2 = self.cell_network2(cell_feat2)

        drug1_feat3=self.drug3MLP1(drug1_3)
        drug2_feat3=self.drug3MLP2(drug2_3)
        cell_feat3=self.cell3MLP(cell_3)

        # cell_feat_vector = cell_feat
        #feat = torch.cat([drug1_feat1_vector, drug1_feat2_vector, drug2_feat1_vector, drug2_feat2_vector, cell_feat_vector, cell_feat_vector2], 1)
        feat1 = torch.cat([drug1_feat1, drug1_feat2, drug1_feat3, drug2_feat1, drug2_feat2, drug2_feat3, cell_feat1, cell_feat2,cell_feat3], 1)
        #feat1 = torch.cat([drug1_feat1, drug2_feat1, cell_feat1, drug1_feat2, drug2_feat2, cell_feat2, drug1_feat3, drug2_feat3,cell_feat3], 1)
        #print(feat1.size())
        out = torch.sigmoid(self.fc_network1(feat1))

        #drug1_feat=torch.cat([drug1_feat1,drug1_feat2,drug1_feat3],dim=1)
        #drug2_feat=torch.cat([drug2_feat1,drug2_feat2,drug2_feat3],dim=1)
        #cell_feat=torch.cat([cell_feat,cell_feat2,cell_feat3],dim=1)
        #drug_feat=torch.max(drug1_feat,drug2_feat)
        #out=torch.sigmoid((drug_feat*cell_feat).sum(dim=1))



        return out
    



class DrugMLP(nn.Module):
    # 默认三层隐藏层，分别有128个 128个 16个神经元
    def __init__(self, input_n=167, output_n=128, num_layer=2, layer_list=[256, 128], dropout=0.5):
        """
        :param input_n: int 输入神经元个数
        :param output_n: int 输出神经元个数
        :param num_layer: int 隐藏层层数
        :param layer_list: list(int) 每层隐藏层神经元个数
        :param dropout: float 训练完丢掉多少
        """
        super(DrugMLP, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.num_layer = num_layer
        self.layer_list = layer_list

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_n, layer_list[0], bias=False),
            nn.ReLU()
        )

        # 隐藏层
        self.hidden_layer = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.ReLU()
        )

    

        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(layer_list[-1], output_n, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output

class CellLineMLP(nn.Module):
    # 默认三层隐藏层，分别有128个 128个 16个神经元
    def __init__(self, input_n, output_n, num_layer=4, layer_list=[8192, 2048,512,128], dropout=0.5):
        """
        :param input_n: int 输入神经元个数
        :param output_n: int 输出神经元个数
        :param num_layer: int 隐藏层层数
        :param layer_list: list(int) 每层隐藏层神经元个数
        :param dropout: float 训练完丢掉多少
        """
        super(CellLineMLP, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.num_layer = num_layer
        self.layer_list = layer_list

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_n, layer_list[0], bias=False),
            nn.ReLU()
        )

        # 隐藏层
        self.hidden_layer = nn.Sequential(
            nn.Linear(8192, 4096, bias=False),
            nn.ReLU(),
            nn.Linear(4096, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU()

        )

        

        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(layer_list[-1], output_n, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output



class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_drugs):
        super(Aggregator, self).__init__()
        self.n_drugs = n_drugs
        

    def forward(self, entity_emb, drug_emb, relation_emb,
                edge_index, edge_type, 
                 disen_weight_att):

        n_entities = entity_emb.shape[0]
        #print(n_entities)
        channel = entity_emb.shape[1]
        n_drugs = self.n_drugs
    

        """KG aggregate"""
        head, tail = edge_index 
        
        edge_relation_emb = relation_emb[edge_type]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)\ 得到每条边对应的embedding [95579,128]
        
        
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [95579,128] 对应公式五
        

        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0) #[25487,128] 合并embedding

        #drug_agg = torch.sparse.mm(interact_mat_drug, entity_emb)
        #print(interact_mat_cell)
        #cell_agg = torch.sparse.mm(interact_mat_cell, entity_emb)
        #print(n_cell)
        #entity_agg[n_drugs:n_drugs+n_cell,]=cell_agg+entity_agg[n_drugs:n_drugs+n_cell,]
        #drug_agg=entity_agg[:n_drugs,]+drug_agg
    
        
        return entity_agg, entity_agg[:n_drugs,],relation_emb



    
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_drugs,
                 n_factors, n_relations,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_drugs = n_drugs
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.lambda_coeff=0.5
        self.topk=5
        #self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact  channel: embding size [n_relations-1,128]
        #print(len(weight))
        
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel] relation embedding

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1)) #公式2 [8,n_relations]
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_drugs=n_drugs))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz() #DTI个数

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        #print(x)
        #print(i)
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    

    def _cul_cor(self):
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        """cul similarity for each latent factor weight pairs"""
        cor = 0
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                if self.ind == 'distance':
                    cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, drug_emb, entity_emb, relation_emb, edge_index, edge_type,
                gpu_id, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout: #对结果造成较大影响
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            #interact_mat_drug = self._sparse_dropout(interact_mat_drug, self.node_dropout_rate)
            #interact_mat_cell = self._sparse_dropout(interact_mat_cell, self.node_dropout_rate)
        
        self.device = torch.device("cuda:" + str(gpu_id))
        #origin_entity_adj = self.build_adj(entity_emb, self.topk)
        entity_res_emb = entity_emb  # [n_entity, channel]
        drug_res_emb = drug_emb  # [n_drugs, channel]
        relation_res_emb=relation_emb
        #cor = self._cul_cor()
        
        for i in range(len(self.convs)):
            entity_emb, drug_emb,relation_emb= self.convs[i](entity_emb, drug_emb, relation_emb,
                                                 edge_index, edge_type, 
                                                  self.disen_weight_att)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                drug_emb = self.dropout(drug_emb)
                relation_emb = self.dropout(relation_emb)
            entity_emb = F.normalize(entity_emb)
            drug_emb = F.normalize(drug_emb)
            relation_emb = F.normalize(relation_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            drug_res_emb = torch.add(drug_res_emb, drug_emb)
            relation_res_emb = torch.add(relation_res_emb, relation_emb)
        
        #entity_adj = (1 - self.lambda_coeff) * self.build_adj(entity_res_emb,
                   #self.topk) + self.lambda_coeff * origin_entity_adj
        #print(entity_res_emb.shape,drug_res_emb.shape)
        return entity_res_emb, drug_res_emb, relation_res_emb
    
    def build_adj(self, context, topk):
        # construct similarity adj matrix
        n_entity = context.shape[0]
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu() #10178X128
        sim = torch.mm(context_norm, context_norm.transpose(1, 0)) #10178X10178 公式2
        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)
        # adj_matrix = (torch.zeros_like(sim)).scatter_(-1, knn_ind, knn_val)
        knn_val, knn_ind = knn_val.to(self.device), knn_ind.to(self.device)

        y = knn_ind.reshape(-1) #铺平
        x = torch.arange(0, n_entity).unsqueeze(dim=-1).to(self.device)
        x = x.expand(n_entity, topk).reshape(-1)
        indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0) #元素的坐标
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_entity, n_entity])).to(self.device)

        # normalized laplacian adj
        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt_value = d_inv_sqrt._values() #公式4
        x = torch.arange(0, n_entity).unsqueeze(dim=0).to(self.device)
        x = x.expand(2, n_entity)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_entity, n_entity]))
        #L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity), d_mat_inv_sqrt)
        a=torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity.to_dense())
        L_norm=torch.sparse.mm(a,d_mat_inv_sqrt.to_dense())
        return L_norm


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph):
        super(Recommender, self).__init__()

        self.n_drugs = data_config['n_drugs']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include cell
        self.n_nodes = data_config['n_nodes']  # n_drugs + n_entities
        #self.drug_two_embed = drug_two_embedding
        #self.dis_two_embed = dis_two_embedding
        
        #self.drug_edge=self.build_graph(self.drug_sim,10)
        #self.dis_edge=self.build_graph(self.dis_sim,10)

        self.decay = args_config.l2 #正则化项系数
        self.sim_decay = args_config.sim_regularity #lamda 1 intent loss 系数
        self.emb_size = args_config.dim #embedding size
        self.context_hops = args_config.context_hops # 3 gcn聚集信息层数
        self.n_factors = args_config.n_factors # intent 个数
        self.node_dropout = args_config.node_dropout #true
        self.node_dropout_rate = args_config.node_dropout_rate # 0.5
        self.mess_dropout = args_config.mess_dropout #true
        self.mess_dropout_rate = args_config.mess_dropout_rate # 0.1
        self.ind = args_config.ind #度量intent相似性的方法
        self.gpu_id=args_config.gpu_id

        self.n_entity_layer=1
        self.lightgcn_layer=2

        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        #self.adj_mat_drug = adj_mat_drug
        #self.adj_mat_cell = adj_mat_cell
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        #self.interact_mat=self._convert_sp_mat_to_sp_tensor(self.adj_mat_drug).to(self.device)
        #self.drug_sim_edge=torch.tensor(drug_sim_edge).to(self.device)
        #self.dis_sim_edge=torch.tensor(dis_sim_edge).to(self.device)
        

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.relation_emb=nn.Parameter(self.relation_emb)

        self.gcn = self._init_model()

        self.fc1 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        
        

        
        


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def build_graph(self, sim, num_neighbor):
        # 根据药物或疾病相似度构造相似度图时，需要人为设置构造好的图中节点的邻居数目，否则构造的图是完全图。
        if num_neighbor>sim.shape[0] or num_neighbor<0:
            num_neighbor = sim.shape[0]
        neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]
        row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
        col_index = neighbor.reshape(-1)    # 展成一维数组
        edge_index = torch.from_numpy(np.array([row_index, col_index]).astype(int))
        values = torch.ones(edge_index.shape[1])
        values = torch.from_numpy(sim[row_index, col_index]).float()*values
        return (edge_index, values, sim.shape)
    
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size)) # 初始化所有节点embedding
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size)) #初始化 intent embedding

        self.relation_emb=initializer(torch.empty(self.n_relations, self.emb_size))
        # [n_drugs, n_nodes]  [n_cell, n_nodes]
        #self.interact_mat_drug = self._convert_sp_mat_to_sp_tensor(self.adj_mat_drug).to(self.device)
        #self.interact_mat_cell = self._convert_sp_mat_to_sp_tensor(self.adj_mat_cell).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_drugs=self.n_drugs,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]
    
    def light_gcn(self, drug_embedding, item_embedding, adj):
        #ego_embeddings = torch.cat((drug_embedding, item_embedding), dim=0)
        ego_embeddings=item_embedding
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        #u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_drugs, self.n_entities-self.n_drugs], dim=0)
        u_g_embeddings=all_embeddings[:self.n_drugs,:]
        i_g_embeddings=all_embeddings
        return u_g_embeddings, i_g_embeddings

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # 返回graph 的边 [h,t,r]
        #print(graph_tensor)
        
        index = graph_tensor[:, :-1]  # [h,t]
        #print(index)
        type = graph_tensor[:, -1]  # relation type
        #print(type)
        return index.t().long().to(self.device), type.long().to(self.device)
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        drug1 = batch['drugs1']
        drug2 = batch['drugs2']
    
        pos_cell = batch['pos_cell']
        #print(pos_cell)
        neg_cell = batch['neg_cell']
        cell=batch['cells'] #1024
        labels=batch['labels']

        drug_emb = self.all_embed[:self.n_drugs, :]
        entity_emb = self.all_embed[:, :]
        relation_emb = self.relation_emb[:, :]

    
        # entity_gcn_emb: [n_entity, channel]
        # drug_gcn_emb: [n_drugs, channel]
        entity_gcn_emb, drug_gcn_emb,relation_gcn_emb = self.gcn(drug_emb,
                                                     entity_emb,
                                                     relation_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.gpu_id,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout
                                                     )
        
        drug1_e_1=drug_gcn_emb[drug1]
        drug2_e_1=drug_gcn_emb[drug2]
        #drug_e_1=torch.max(drug1_e_1,drug2_e_1)

        cell_e_1=relation_gcn_emb[cell]
        




       
        
        

        




        

        #return self.new_create_bpr_loss(drug_e,cell_e, 1,0)
        return self.create_bce_loss(drug1_e_1,drug2_e_1,cell_e_1,labels,0)
    
    def get_ent_rel_emb(self):
        return self.all_embed,self.relation_emb
        

    def generate(self,drug1,drug2,cell):
        #drug = batch['drugs']
        #pos_cell = batch['pos_cell']
        #print(pos_cell)
        #neg_cell = batch['neg_cell']
        #cell=batch['cell'] #1024
        #labels=batch['labels']


        drug_emb = self.all_embed[:self.n_drugs, :]
        entity_emb = self.all_embed[:, :]
        relation_emb= self.relation_emb[:, :]
        # entity_gcn_emb: [n_entity, channel]
        # drug_gcn_emb: [n_drugs, channel]
        entity_gcn_emb, drug_gcn_emb,relation_gcn_emb = self.gcn(drug_emb,
                                                     entity_emb,
                                                     relation_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.gpu_id,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout
                                                     )
        #print(drug_gcn_emb==entity_gcn_emb[:self.n_drugs,])

        #drug_sim_emb = self.all_embed[:self.n_drugs, :]
        #dis_sim_emb = self.all_embed[self.n_drugs:self.n_drugs+self.n_cell, :]

        #drug_sim_emb=self.drug_sim_encoder(drug_sim_emb,self.drug_sim_edge,np.array(range(len(self.drug_sim_edge))),p=0.5)
        #dis_sim_emb=self.dis_sim_encoder(dis_sim_emb,self.drug_sim_edge,np.array(range(len(self.dis_sim_edge))),p=0.5)

        drug1_e_1=drug_gcn_emb[drug1]
        drug2_e_1=drug_gcn_emb[drug2]
        cell_e_1=relation_gcn_emb[cell]
        #drug_e_1=torch.max(drug1_e_1,drug2_e_1)

        #interact_mat_new = self.interact_mat
        #print(interact_mat_new)
        #indice_old = interact_mat_new._indices()
        #value_old = interact_mat_new._values()
        #x = indice_old[0, :]
        #y = indice_old[1, :]
        #x_A = x
        #y_A = y + self.n_drugs
        #x_A_T = y + self.n_drugs
        #y_A_T = x
        #x_new = torch.cat((x_A, x_A_T), dim=-1)
        #y_new = torch.cat((y_A, y_A_T), dim=-1)
        #indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        #value_new = torch.cat((value_old, value_old), dim=-1)
        #interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size([self.n_entities, self.n_entities]))
        #drug_lightgcn_emb, item_lightgcn_emb = self.light_gcn(drug_gcn_emb, entity_gcn_emb, interact_graph)
        #drug1_e_2 = drug_lightgcn_emb[drug1]
        #drug2_e_2 = drug_lightgcn_emb[drug2]
        #cell_e_2 = item_lightgcn_emb[cell]

        #drug_e_2=torch.max(drug1_e_2,drug2_e_2)

        #i_h=entity_gcn_emb
        #for i in range(self.n_entity_layer):
            #i_h = torch.sparse.mm(entity_adj, i_h)
        #i_h = F.normalize(i_h, p=2, dim=1)

        #drug1_e_3=i_h[drug1]
        #drug2_e_3=i_h[drug2]
        #cell_e_3=i_h[cell]

        #drug_e_3=torch.max(drug1_e_3,drug2_e_3)
       

        

        #feat=torch.cat([drug1_e_1,drug2_e_1,cell_e_1],1)
        #out=self.fc_network(feat)


        #scores = torch.sum(torch.mul(drug_e, cell_e), axis=1)
        

        return (drug1_e_1*cell_e_1*drug2_e_1).sum(dim=-1)
        

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t()) #产生结果概率
    
    def regularization(self):
        regularization = 0
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        return regularization

    def create_bpr_loss(self, drugs, pos_cell, neg_cell, cor):
        
        batch_size = drugs.shape[0]
        print(drugs)
        pos_scores = torch.sum(torch.mul(drugs, pos_cell), axis=1)
        
        neg_scores = torch.sum(torch.mul(drugs, neg_cell), axis=1)
        #print(batch_size,pos_scores,neg_scores)
        #print(drugs.shape,pos_cell.shape)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))# BPR loss

        # cul regularizer
        regularizer = (torch.norm(drugs) ** 2
                       + torch.norm(pos_cell) ** 2
                       + torch.norm(neg_cell) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor#intent loss

        return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor
    
    def calculate_loss(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        # refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        # between_sim_1 = f(self.sim(B_embedding, A_embedding))
        # loss_2 = -torch.log(
        #     between_sim_1.diag()
        #     / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        # ret = (loss_1 + loss_2) * 0.5
        ret = loss_1
        ret = ret.mean()
        return ret
    def toxic(self, drug1_embeddings, drug2_embeddings):
        return (drug1_embeddings * drug2_embeddings).sum(dim=1)
    def create_bce_loss(self, drugs1,drugs2, cells, labels,loss):
        batch_size = drugs1.shape[0]
        #print(items.size())
        #feat=torch.cat([drugs1,drugs2,cells],1)
        out=(drugs1*cells*drugs2).sum(dim=-1)
        #scores = torch.sum(torch.mul(drugs1, items), axis=1)
        #scores = torch.sigmoid(score)
        criteria = nn.BCEWithLogitsLoss()
        bce_loss = criteria(out, labels.float())
        #print(bce_loss)
        # cul regularizer
        regularizer = (torch.norm(drugs1) ** 2
                       + torch.norm(drugs2) ** 2
                       +torch.norm(cells) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * loss
        return bce_loss+emb_loss+cor_loss  , bce_loss, emb_loss
    
   

    
    
    def new_create_bpr_loss(self, drugs, cell, score,loss):
        batch_size = int(drugs.shape[0]/2)
        #print(batch_size)
        pos_scores = torch.sum(torch.mul(drugs[:batch_size], cell[:batch_size]), axis=1)
        
        neg_scores = torch.sum(torch.mul(drugs[batch_size:], cell[batch_size:]), axis=1)
        #print(batch_size,pos_scores,neg_scores)
        #print(drugs.shape,pos_cell.shape)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))# BRP loss

        # cul regularizer
        regularizer = (torch.norm(drugs) ** 2
                       + torch.norm(cell) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * loss#intent loss

        return mf_loss+emb_loss+cor_loss , mf_loss, 0
