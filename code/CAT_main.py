import gc
import os
import sys
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from EarlyStopping import *
from utils import setup_seed
import copy
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import LinkNeighborLoader
from torch_scatter import scatter_min
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
import time
from CAT_model import cat_model
from collections import Counter
import dgl
from torch_geometric.data import Data, HeteroData
import argparse


def parameter_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trust Prediction")

    # Model parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GNN layers')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='Negative slope for LeakyReLU')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # Data parameters
    parser.add_argument('--val_ratio', type=float, default=0.7, help='Validation time ratio')
    parser.add_argument('--test_ratio', type=float, default=0.85, help='Test time ratio')
    parser.add_argument('--mask_ratio', type=float, default=0.0, help='Ratio of nodes to mask')
    parser.add_argument('--neg_seed', type=int, default=2023, help='Negative sampling seed')

    # Neighbor sampling parameters
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[30, 10], help='Number of neighbors to sample at each layer')
    parser.add_argument('--neg_sampling_ratio', type=float, default=1.0, help='Negative sampling ratio')

    # Path parameters
    parser.add_argument('--data_path', type=str, default='/data/Epinions', help='Path to data directory')
    parser.add_argument('--model_save_path', type=str, default='/models/cat.pkl', help='Path to save model')

    # Other parameters
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    return parser.parse_args()


root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)


def get_batch_node_time_dict(data, edge_label_index, edge_label_time):
    """For edges in a batch replace `src` and `dst` node times by the min
    across all edge times."""

    def update_time_(node_time_dict, index, node_type, num_nodes):
        node_time_dict[node_type] = node_time_dict[node_type].clone()
        node_time, _ = scatter_min(edge_label_time, index, dim=0, dim_size=num_nodes)  # Get the minimum timestamp for each node and sort in order (from small to large)
        # NOTE We assume that node_time is always less than edge_time.
        index_unique = index.unique()
        count = 0
        if data.node_types[0] == data.node_types[1]:
            if count == 1:
                node_time_dict_ori = node_time_dict[node_type]
                node_time_dict[node_type][index_unique] = node_time[index_unique]
                min_values = torch.minimum(node_time_dict_ori, node_time_dict)
                # Replace zeros with larger values
                node_time_dict[node_type] = torch.where(min_values == 0, torch.maximum(node_time_dict_ori, node_time_dict), min_values)
                count = count + 1
        else:
            node_time_dict[node_type][index_unique] = node_time[index_unique]

    node_time_dict = copy.copy(data.time_dict)
    num_src_nodes = data[data.node_types[0]].num_nodes
    num_dst_nodes = data[data.node_types[-1]].num_nodes
    update_time_(node_time_dict, edge_label_index[0], data.node_types[0], num_src_nodes)  # for user
    update_time_(node_time_dict, edge_label_index[1], data.node_types[-1], num_dst_nodes)  # for item

    return node_time_dict


def get_ori_graph(args):
    """Reading original graph from files."""
    graph = HeteroData()
    trust_with_time = np.loadtxt(root_path + args.data_path + '/user_user_time.txt').astype(np.int64)
    user_user = torch.from_numpy(trust_with_time[:, 0:2].T)
    user_user_time = torch.from_numpy(trust_with_time[:, 2])
    user_user_node_time = torch.from_numpy(trust_with_time[:, :])

    # user, item, item category(1-25), rating, helpfulness, date
    rating_with_time = np.genfromtxt(root_path + args.data_path + '/user_item_time.txt', delimiter=',').astype(np.int64)
    user_item = torch.from_numpy(rating_with_time[:, :2].T)
    user_item_time = torch.from_numpy(rating_with_time[:, 5].T)

    # item category
    df_type = pd.DataFrame(rating_with_time[:, 1:3], columns=['item', 'type'])
    item_type_groups = df_type.groupby('item')['type'].apply(list)  # Group by item and get list of types for each item
    most_common_type = {}  # Each item uniquely corresponds to one category
    # Calculate the most common type for each item
    for item, types in item_type_groups.items():
        most_common_type[item] = Counter(types).most_common(1)[0][0]
    items = np.array(list(most_common_type.keys()))
    types = np.array(list(most_common_type.values()))
    item_type = torch.from_numpy(np.vstack((items, types)))

    user_max = int(max(user_user[0].max(), user_user[1].max())) + 1
    item_max = int(rating_with_time[:,1].max()) + 1

    graph['user'].x = torch.randn(user_max, args.emb_size)
    graph['item'].x = torch.randn(item_max, args.emb_size)

    graph["user"].time = torch.zeros(graph["user"].x.size(0)).long()
    graph["item"].time = torch.zeros(graph["item"].x.size(0)).long()
    graph["user", "to", "user"].edge_index = user_user
    graph["user", "to", "user"].edge_label_time = user_user_time
    graph["user", "to", "item"].edge_index = user_item
    graph["user", "to", "item"].edge_label_time = user_item_time
    graph["item", "to", "user"].edge_index = user_item.flip(0)
    graph["item", "to", "user"].edge_label_time = user_item_time

    graph_user_item_time_dict = get_batch_node_time_dict(graph, graph["user", "to", "item"].edge_index, graph["user", "to", "item"].edge_label_time)
    graph["item"].time = graph_user_item_time_dict["item"]
    graph["user"].time = graph_user_item_time_dict["user"]
    # Update minimum timestamp based on user-user time
    user_node_time = torch.cat((user_user_node_time[:, [0, 2]], user_user_node_time[:, [1, 2]]), dim=0).T
    # Find the minimum time for each node
    node_time, _ = scatter_min(user_node_time[1, :], user_node_time[0, :], dim=0, dim_size=graph["user"].num_nodes)
    min_user_time = torch.minimum(graph["user"].time, node_time)
    # May be because some users have no user-item interaction. If timestamp is 0, replace with the user-item corresponding time.
    graph["user"].time = torch.where(min_user_time == 0, torch.maximum(graph["user"].time, node_time), min_user_time)

    print("Graph", graph, '\n')

    return graph, item_type


def get_dataset(data_classif, graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set):
    """Split the dataset into train, val, and test set."""
    data = HeteroData()
    data["user"].x = graph["user"].x
    data["user"].node_time = graph["user"].time
    data["item"].x = graph["item"].x
    data["item"].node_time = graph["item"].time

    if data_classif == "val" or data_classif == "test":
        data["user", "to", "item"].edge_index = graph["user", "to", "item"].edge_index
        data["user", "to", "item"].edge_label_time = graph["user", "to", "item"].edge_label_time
        data["item", "to", "user"].edge_index = graph["item", "to", "user"].edge_index
        data["item", "to", "user"].edge_label_time = graph["item", "to", "user"].edge_label_time

    if data_classif == "train":
        data["user", "to", "user"].edge_index = graph["user", "to", "user"].edge_index[:, valid_train_flag]
        data["user", "to", "user"].edge_label_index = graph["user", "to", "user"].edge_index[:, valid_train_flag]
        data["user", "to", "user"].edge_label = torch.ones(np.sum(valid_train_flag))
        data["user", "to", "user"].edge_label_time = graph["user", "to", "user"].edge_label_time[valid_train_flag]
        # Mask users in user-item
        mask = torch.isin(graph["user", "to", "item"].edge_index[0], torch.tensor(mask_node_set))
        data["user", "to", "item"].edge_index = graph["user", "to", "item"].edge_index[:, ~mask]
        data["user", "to", "item"].edge_label_time = graph["user", "to", "item"].edge_label_time[~mask]
        data["item", "to", "user"].edge_index = data["user", "to", "item"].edge_index.flip(0)
        data["item", "to", "user"].edge_label_time = graph["item", "to", "user"].edge_label_time[~mask]
        # Delete user-item interactions greater than val_time
        time_flag = data["user", "to", "item"].edge_label_time < val_time
    elif data_classif == "val":
        data["user", "to", "user"].edge_index = graph["user", "to", "user"].edge_index[:, valid_train_flag]
        data["user", "to", "user"].edge_label_index = graph["user", "to", "user"].edge_index[:, valid_val_flag]
        data["user", "to", "user"].edge_label = torch.ones(np.sum(valid_val_flag))
        data["user", "to", "user"].edge_label_time = graph["user", "to", "user"].edge_label_time[valid_train_flag]
        # Delete user-item interactions greater than val_time, same with train_data
        time_flag = data["user", "to", "item"].edge_label_time < val_time
    else:
        data["user", "to", "user"].edge_index = torch.cat((graph["user", "to", "user"].edge_index[:, valid_train_flag], graph["user", "to", "user"].edge_index[:, valid_val_flag]), dim=1)
        data["user", "to", "user"].edge_label_index = graph["user", "to", "user"].edge_index[:, valid_test_flag]
        data["user", "to", "user"].edge_label = torch.ones(np.sum(valid_test_flag))
        data["user", "to", "user"].edge_label_time = torch.cat((graph["user", "to", "user"].edge_label_time[valid_train_flag], graph["user", "to", "user"].edge_label_time[valid_val_flag]), dim=0)
        # Delete user-item interactions greater than test_time
        time_flag = data["user", "to", "item"].edge_label_time < test_time

    # Make user-item interactions conform to time order
    data["user", "to", "item"].edge_index = data["user", "to", "item"].edge_index[:, time_flag]
    data["user", "to", "item"].edge_label_time = data["user", "to", "item"].edge_label_time[time_flag]
    data["item", "to", "user"].edge_index = data["item", "to", "user"].edge_index[:, time_flag]
    data["item", "to", "user"].edge_label_time = data["item", "to", "user"].edge_label_time[time_flag]

    return data


def negative_sampler(data, num_nodes, seed):
    random.seed(seed)
    neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([data.edge_label, data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

    return edge_label, edge_label_index


def get_metrics(out, edge_label):
    edge_label = edge_label.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    pred = np.argmax(out, axis=1)

    auc = roc_auc_score(edge_label, out[:, 1])
    acc = accuracy_score(edge_label, pred)
    f1 = f1_score(edge_label, pred)
    ap = average_precision_score(edge_label, out[:, 1])

    return auc, acc, f1, ap


def to_dgl(data):
    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()

        g = dgl.graph((row, col))

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, edge_store in data.edge_items():
            if edge_store.get('edge_index') is not None:
                row, col = edge_store.edge_index
            else:
                row, col, _ = edge_store['adj_t'].t().coo()

            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, node_store in data.node_items():
            for attr, value in node_store.items():
                if attr == 'x' or attr == 'node_time':  # we only need the feature tensor
                    g.nodes[node_type].data[attr] = value
                else:
                    continue

        # dealing with other edge-related attributes, such as edge_label, edge_label_index, edge_label_time
        for edge_type, edge_store in data.edge_items():
            for attr, value in edge_store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                if attr == 'edge_time':
                    g.edges[edge_type].data[attr] = value
                else:
                    continue

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")


def creat_edge_time(data):
    """create edge gap time for each edge."""
    for edge_type, edge_store in data.edge_items():
        dst_type = edge_type[-1]
        data[edge_type].edge_time = (-(data[dst_type].node_time[data[edge_type].edge_index[1]] - data[edge_type].edge_label_time)).float()
        # print("max and min time span:", data[edge_type].edge_time.max().item(), data[edge_type].edge_time.min().item())

    return data


def mrr(y_pred_pos, y_pred_neg):
    y_pred_pos = y_pred_pos.reshape(-1, 1)
    # optimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
    # pessimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1. / ranking_list
    return mrr_list.mean()


@torch.no_grad()
def test(model, data, new_node_set, item_type, device):
    model.eval()

    # cal loss
    criterion = torch.nn.CrossEntropyLoss().to(device)

    tr = data['user', 'to', 'user'].edge_label_index[0].detach().cpu().numpy()
    te = data['user', 'to', 'user'].edge_label_index[1].detach().cpu().numpy()
    new_flag = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(tr, te)])
    old_flag = ~new_flag

    edge_label_index_old = data['user', 'to', 'user'].edge_label_index[:, old_flag].to(device)
    edge_label_old = data['user', 'to', 'user'].edge_label[old_flag].to(device)
    edge_label_index_new = data['user', 'to', 'user'].edge_label_index[:, new_flag].to(device)
    edge_label_new = data['user', 'to', 'user'].edge_label[new_flag].to(device)

    hg = to_dgl(data).to(device)

    item_id = torch.arange(1, data['item'].num_nodes)
    out_old = model(hg, hg.ndata['x'], edge_label_index_old, item_type, item_id)
    loss_old = criterion(out_old, edge_label_old.long())

    out_new = model(hg, hg.ndata['x'], edge_label_index_new, item_type, item_id)
    loss_new = criterion(out_new, edge_label_new.long())
    # print("check", np.sum(old_flag), np.sum(new_flag))

    out_old = torch.nn.functional.softmax(out_old, dim=1)
    out_new = torch.nn.functional.softmax(out_new, dim=1)

    # cal mrr
    neg_idx_old = torch.where(edge_label_old == 0)[0][0].item()  # Index of the first negative sample
    y_pred_pos_old = out_old[:neg_idx_old, ][:, 1].detach().cpu().numpy()  # Probability that positive samples are predicted as positive samples
    y_pred_neg_old = out_old[neg_idx_old:, ][:, 1].detach().cpu().numpy()  # Probability that negative samples are predicted as positive samples
    # From the negative sample prediction probability sequence, randomly select k negative sample prediction probabilities for each positive sample
    idx_old = np.random.randint(0, len(y_pred_neg_old), size=(len(y_pred_pos_old), 100))
    mrr_old = mrr(y_pred_pos_old, y_pred_neg_old[idx_old])

    neg_idx_new = torch.where(edge_label_new == 0)[0][0].item()  # Index of the first negative sample
    y_pred_pos_new = out_new[:neg_idx_new, ][:, 1].detach().cpu().numpy()  # Probability that positive samples are predicted as positive samples
    y_pred_neg_new = out_new[neg_idx_new:, ][:, 1].detach().cpu().numpy()  # Probability that negative samples are predicted as positive samples
    # From the negative sample prediction probability sequence, randomly select k negative sample prediction probabilities for each positive sample
    idx_new = np.random.randint(0, len(y_pred_neg_new), size=(len(y_pred_pos_new), 100))
    mrr_new = mrr(y_pred_pos_new, y_pred_neg_new[idx_new])

    # cal metrics
    auc_old, acc_old, f1_old, ap_old = get_metrics(out_old, edge_label_old)
    auc_new, acc_new, f1_new, ap_new = get_metrics(out_new, edge_label_new)

    model.train()

    return loss_old.item(), auc_old, acc_old, f1_old, ap_old, mrr_old, auc_new, acc_new, f1_new, ap_new, mrr_new


def train(model, train_loader, val_data, test_data, new_node_set, item_type, args):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    time_list = []
    max_val_ap = 0
    final_test_auc_old = 0
    final_test_acc_old = 0
    final_test_f1_old = 0
    final_test_ap_old = 0
    final_test_mrr_old = 0
    final_test_auc_new = 0
    final_test_acc_new = 0
    final_test_f1_new = 0
    final_test_ap_new = 0
    final_test_mrr_new = 0
    best_epoch = 0
    model.train()
    for epoch in tqdm(range(args.epochs)):
        total_loss = total_samples = 0
        time_list_batch = []
        for batch_index, sampled_train_data in enumerate(train_loader):
            edge_label_index = sampled_train_data["user", "to", "user"].edge_label_index.to(args.device)
            edge_label = sampled_train_data["user", "to", "user"].edge_label.to(args.device)
            hg = to_dgl(sampled_train_data).to(args.device)

            start_time = time.time()
            optimizer.zero_grad()
            out = model(hg, hg.ndata['x'], edge_label_index, item_type, sampled_train_data['item'].n_id)
            loss = criterion(out, edge_label.long())
            loss.backward()
            optimizer.step()
            time_list_batch.append(time.time() - start_time)

            total_loss += loss.item() * out.shape[0]
            total_samples += out.shape[0]
            if (batch_index + 1) % 100 == 0:
                print("******->", batch_index + 1, total_loss / total_samples)

            torch.cuda.empty_cache()  # force the gpu to release the unuse memory ASAP
            gc.collect()

        print("time for each epoch:", np.sum(time_list_batch))
        time_list.append(np.sum(time_list_batch))

        # validation
        val_loss, val_auc_old, val_acc_old, val_f1_old, val_ap_old, val_mrr_old, val_auc_new, val_acc_new, val_f1_new, val_ap_new, val_mrr_new \
            = test(model, val_data, new_node_set, item_type, args.device)
        test_loss, test_auc_old, test_acc_old, test_f1_old, test_ap_old, test_mrr_old, test_auc_new, test_acc_new, test_f1_new, test_ap_new, test_mrr_new \
            = test(model, test_data, new_node_set, item_type, args.device)

        if val_ap_old > max_val_ap:
            best_epoch = epoch + 1
            max_val_ap = val_ap_old
            final_test_auc_old = test_auc_old
            final_test_acc_old = test_acc_old
            final_test_f1_old = test_f1_old
            final_test_ap_old = test_ap_old
            final_test_mrr_old = test_mrr_old
            final_test_auc_new = test_auc_new
            final_test_acc_new = test_acc_new
            final_test_f1_new = test_f1_new
            final_test_ap_new = test_ap_new
            final_test_mrr_new = test_mrr_new
            best_model = copy.deepcopy(model)
            # save model
            state = {'model': best_model.state_dict()}
            torch.save(state, root_path + args.model_save_path)

        early_stopping(val_ap_old, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} test_loss {:.8f}'.format(epoch + 1, total_loss / total_samples, val_loss, test_loss))
        print('val on old nodes')
        print('auc {:.4f} acc {:.4f} f1 {:.4f} ap {:.4f} mrr {:.4f}'.format(val_auc_old, val_acc_old, val_f1_old, val_ap_old, val_mrr_old))
        print('val on new nodes')
        print('auc {:.4f} acc {:.4f} f1 {:.4f} ap {:.4f} mrr {:.4f}'.format(val_auc_new, val_acc_new, val_f1_new, val_ap_new, val_mrr_new))
        print('test on old nodes')
        print('auc {:.4f} acc {:.4f} f1 {:.4f} ap {:.4f} mrr {:.4f}'.format(test_auc_old, test_acc_old, test_f1_old, test_ap_old, test_mrr_old))
        print('test on new nodes')
        print('auc {:.4f} acc {:.4f} f1 {:.4f} ap {:.4f} mrr {:.4f}'.format(test_auc_new, test_acc_new, test_f1_new, test_ap_new, test_mrr_new))


    return (final_test_auc_old, final_test_acc_old, final_test_f1_old, final_test_ap_old, final_test_mrr_old,
            final_test_auc_new, final_test_acc_new, final_test_f1_new, final_test_ap_new, final_test_mrr_new,
            best_epoch, np.mean(time_list))


def main():
    args = parameter_parser()

    # Setup
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Load graph
    graph, item_type = get_ori_graph(args)
    trustor = np.array(graph["user", "to", "user"].edge_index[0])
    trustee = np.array(graph["user", "to", "user"].edge_index[1])
    ts = np.array(graph["user", "to", "user"].edge_label_time)
    assert (len(trustor) == len(trustee) == len(ts))
    val_time, test_time = list(np.quantile(ts, [args.val_ratio, args.test_ratio]))
    total_node_set = set(np.unique(np.hstack([trustor, trustee])))

    # masking users if needed
    mask_node_set = random.sample(list(set(trustor[ts > val_time]).union(set(trustee[ts > val_time]))), int(args.mask_ratio * len(total_node_set)))  # list

    mask_tr_flag = np.isin(trustor, mask_node_set)  # Calculate whether trustor is in the mask set
    mask_te_flag = np.isin(trustee, mask_node_set)
    none_node_flag = (1 - mask_tr_flag) * (1 - mask_te_flag)  # Only keep edges where both trustor and trustee are not in mask set

    valid_train_flag = (ts <= val_time) * (none_node_flag > 0)  # Edges that are really used for training
    train_trustor = trustor[valid_train_flag]
    train_trustee = trustee[valid_train_flag]

    # define the new node sets for testing inductiveness of the model
    train_node_set = set(train_trustor).union(train_trustee)  # Training set node set
    assert (len(train_node_set - set(mask_node_set)) == len(train_node_set))  # Ensure that nodes in the mask set do not appear in the training set
    new_node_set = total_node_set - train_node_set
    print("#new nodes:", len(new_node_set))

    # select validation and test dataset
    valid_val_flag = (ts > val_time) * (ts <= test_time)
    valid_test_flag = ts > test_time
    print("valid train/val/test:", np.sum(valid_train_flag), np.sum(valid_val_flag), np.sum(valid_test_flag))

    # get train/valid/test data
    train_data = get_dataset("train", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
    val_data = get_dataset("val", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
    test_data = get_dataset("test", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
    train_data = creat_edge_time(train_data)
    val_data = creat_edge_time(val_data)
    test_data = creat_edge_time(test_data)

    # get pre-trained embeddings using metapath2vec
    # user_emb, item_emb = get_embedding(args.emb_size, train_data, val_data, test_data, item_type, [("user", "to", "item"), ("item", "to", "type"), ("type", "to", "item"), ("item", "to", "user")], device)
    user_emb = torch.from_numpy(np.loadtxt(root_path + '/heterogeneous/uitiu_user_emb_meta_{}.txt'.format(args.emb_size))).type(torch.float32)
    item_emb = torch.from_numpy(np.loadtxt(root_path + '/heterogeneous/uitiu_item_emb_meta_{}.txt'.format(args.emb_size))).type(torch.float32)
    train_data["user"].x = val_data["user"].x = test_data["user"].x = user_emb
    train_data["item"].x = val_data["item"].x = test_data["item"].x = item_emb
    print("user_emb:", user_emb.shape, "item_emb:", item_emb.shape, "\n")

    print("train", train_data, "\n")
    print("val", val_data, "\n")
    print("test", test_data, "\n")

    # negative sampling for validation and test sets
    edge_label_val, edge_label_index_val = negative_sampler(val_data["user", "to", "user"], val_data["user"].num_nodes, args.neg_seed)
    val_data["user", "to", "user"].edge_label_index = edge_label_index_val
    val_data["user", "to", "user"].edge_label = edge_label_val

    edge_label_test, edge_label_index_test = negative_sampler(test_data["user", "to", "user"], test_data["user"].num_nodes, args.neg_seed + 1)
    test_data["user", "to", "user"].edge_label_index = edge_label_index_test
    test_data["user", "to", "user"].edge_label = edge_label_test

    # get mini-batch for train data
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=args.num_neighbors,
        edge_label_index=(("user", "to", "user"), train_data["user", "to", "user"].edge_label_index),
        edge_label=train_data["user", "to", "user"].edge_label,
        edge_label_time=train_data["user", "to", "user"].edge_label_time,
        time_attr="node_time",
        temporal_strategy="last",  # Take the node closest to the target time
        neg_sampling_ratio=args.neg_sampling_ratio,  # negative edges differ for every training epoch
        batch_size=args.batch_size,
        shuffle=True  # Shuffle data to increase randomness
    )

    # Create model and train
    model = cat_model(
        num_layers=args.num_layers,
        in_dim=args.emb_size,
        hidden_dim=args.hidden_dim,
        ntypes=["user", "item"],
        negative_slope=args.negative_slope,
        seed=args.seed
    ).to(device)

    # for training
    test_auc_old, test_acc_old, test_f1_old, test_ap_old, test_mrr_old, test_auc_new, test_acc_new, test_f1_new, test_ap_new, test_mrr_new, best_epoch, mean_time \
        = train(model, train_loader, val_data, test_data, new_node_set, item_type, args)
    print('final best epoch:', best_epoch)
    print('mean time for each epoch:', mean_time)

    # for testing only
    # model.load_state_dict(torch.load(root_path + '/models/cat.pkl')['model'])
    # test_loss, test_auc_old, test_acc_old, test_f1_old, test_ap_old, test_mrr_old, test_auc_new, test_acc_new, test_f1_new, test_ap_new, test_mrr_new \
    #     = test(model, test_data, new_node_set, item_type)

    print('final best for old nodes:')
    print('auc {:.4f} acc {:.4f} f1 {:.4f} ap {:.4f} mrr {:.4f}'.format(test_auc_old, test_acc_old, test_f1_old, test_ap_old, test_mrr_old))
    print('final best for new nodes:')
    print('auc {:.4f} acc {:.4f} f1 {:.4f} ap {:.4f} mrr {:.4f}'.format(test_auc_new, test_acc_new, test_f1_new, test_ap_new, test_mrr_new))


if __name__ == '__main__':
    main()