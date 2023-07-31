import argparse
import copy
import os.path
import sys

sys.path.append("./")

import torch
import time
from sklearn import metrics
from torch.optim import Adam
from torch_geometric.datasets import Amazon
from model_with_dirichlet import MGIGNN
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from torch_geometric.transforms import RandomNodeSplit
import torch_geometric.transforms as T
from utils import find_common_neighbors_matrix, count_degree
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import timeit


cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def train_test(run_time, args):
    log_dir = args.log_dir
    sim_function = args.sim_function
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    device = torch.device("cuda:{}".format(str(args.cuda_id)) if torch.cuda.is_available()
                                                                 and args.cuda else 'cpu')
    enable_different_memorysize = args.enable_different_memorysize
    print(device)

    if 'gcnii' not in args.model:
        path = "./data/Amazon"
        dataset = Amazon(path, name='Computers')
        transform = RandomNodeSplit(split='train_rest',
                                    num_val=5500, num_test=5500)
        computer_data = transform(dataset[0])
        computer_data = computer_data.to(device, 'x', 'y')

        train_edge_index = subgraph(computer_data.train_mask,
                                    computer_data.edge_index,
                                    relabel_nodes=True)[0]
        train_x = computer_data.x[computer_data.train_mask]
        train_y = computer_data.y[computer_data.train_mask]
        train_data = Data(edge_index=train_edge_index,
                          x=train_x,
                          y=train_y)
        train_data.num_nodes = train_x.shape[0]

        kwargs = {'batch_size': 1024, 'num_workers': 6,
                  'persistent_workers': True}
        train_loader = NeighborLoader(train_data,
                                      input_nodes=None,
                                      num_neighbors=[25, 10], shuffle=True,
                                      **kwargs)
        train_loader.data.num_nodes = train_data.num_nodes
        train_loader.data.n_id = torch.arange(train_data.num_nodes)

        subgraph_loader = NeighborLoader(copy.copy(computer_data),
                                         input_nodes=None,
                                         num_neighbors=[-1], shuffle=False,
                                         **kwargs)
        subgraph_loader.data.num_nodes = computer_data.num_nodes
        subgraph_loader.data.n_id = torch.arange(computer_data.num_nodes)


    elif 'gcnii' in args.model:
        path = "./data/GCN2_Amazon"
        pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
        dataset = Amazon(path, name='Computers',
                         pre_transform=pre_transform)
        transform = RandomNodeSplit(split='train_rest',
                                    num_val=5500, num_test=5500)
        computer_data = transform(dataset[0])
        computer_data = computer_data.to(device, 'x', 'y')

        row, col, edge_attr = computer_data.adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        train_edge_index = subgraph(computer_data.train_mask,
                                    edge_index=edge_index,
                                    relabel_nodes=True)[0]
        train_x = computer_data.x[computer_data.train_mask]
        train_y = computer_data.y[computer_data.train_mask]
        train_data = Data(edge_index=train_edge_index,
                          x=train_x,
                          y=train_y)
        train_data = pre_transform(train_data)
        train_data.num_nodes = train_x.shape[0]

        kwargs = {'batch_size': 1024, 'num_workers': 6,
                  'persistent_workers': True}
        train_loader = NeighborLoader(train_data,
                                      input_nodes=None,
                                      num_neighbors=[25, 10], shuffle=True,
                                      **kwargs)
        train_loader.data.num_nodes = train_data.num_nodes
        train_loader.data.n_id = torch.arange(train_data.num_nodes)

        subgraph_loader = NeighborLoader(copy.copy(computer_data),
                                         input_nodes=None,
                                         num_neighbors=[-1], shuffle=False,
                                         **kwargs)
        subgraph_loader.data.num_nodes = computer_data.num_nodes
        subgraph_loader.data.n_id = torch.arange(computer_data.num_nodes)

    print("|> Length of train_loader ", len(train_loader))

    if args.criterion == 'sigmoid':
        criterion = torch.nn.BCELoss()
    elif args.criterion == 'softmax':
        criterion = torch.nn.CrossEntropyLoss()

    num_nodes = computer_data.x.shape[0]

    # Define the model
    model = MGIGNN(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        nodes_numbers=computer_data.x.shape[0],
        normalize=args.normalize,
        k=args.k,
        eta=args.eta,
        device=device,
        model_name=args.model,
        criterion=args.criterion).to(device)
    print(model)

    embedding_optimizer = Adam(model.embedding_encoder.parameters(), lr=args.gnn_lr)
    nodevae_optimizer = Adam(list(model.node_vae.parameters()), lr=args.vae_lr)

    # start_time = time.time()
    bad_counter = 0
    least_loss = float('inf')

    print("|> Start Pretraining.")
    for epoch in range(50):
        model.train()
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:04d}')

        for batch in train_loader:
            batch = batch.to(device)
            embedding_optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            output = model.pretrain_forward(batch)
            loss = criterion(output, y)
            loss.backward()
            embedding_optimizer.step()

            pbar.update(batch.batch_size)

    print("|> End Pretraining.")

    print("|> Start EM Update.")
    for epoch in range(args.epochs):

        model.train()
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:04d}')
        total_loss = 0.0

        for batch in train_loader:
            # E Step
            model.embedding_encoder.requires_grad = False
            batch = batch.to(device)
            nodevae_optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            if sim_function == 'feature_base':
                neighbors_info = None
                train_info = model.get_training_embedding_memory(
                    batch,
                    neighbors_info=neighbors_info
                )
            elif sim_function == 'common_neighbor':
                neighbors_info = find_common_neighbors_matrix(
                    batch,
                    num_nodes=num_nodes,
                    device=device,
                    model_name=args.model
                )
                train_info = model.get_training_embedding_memory(
                    batch,
                    neighbors_info=neighbors_info
                )
            elif sim_function == 'degree':
                neighbors_info = count_degree(
                    batch,
                    num_nodes=num_nodes,
                    device=device,
                    model_name=args.model
                )
                train_info = model.get_training_embedding_memory(
                    batch,
                    neighbors_info=neighbors_info
                )

            if sim_function == 'feature_base':
                sim_info = model.get_sim_info(
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_batch_id'],
                    None
                )
            elif sim_function == 'common_neighbor':
                sim_info = model.get_sim_info(
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_batch_id'],
                    train_info['neighbors_info']
                )
            elif sim_function == 'degree':
                sim_info = model.get_sim_info(
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_batch_id'],
                    train_info['neighbors_info']
                )


            output = model.reconstruct_z_q(train_info, sim_info)

            loss_e = criterion(output, y) + 0.01 * model.node_vae.compute_KL() + 0.01 * model.compute_kl_theta()
            total_loss += loss_e.item()
            loss_e.backward(retain_graph=True)
            nodevae_optimizer.step()

            model.update_lambda_stats_sum(sim_info)
            model.distributed_update_lambda(
                lambda_stats_sum=model.lambda_stats_sum,
                num_of_nodes=model.batch_size,
                update_num=model._num_updates)

            model._num_updates += 1

            # M Step
            model.embedding_encoder.requires_grad = True
            model.node_vae.requires_grad = False
            embedding_optimizer.zero_grad()
            if sim_function == 'feature_base':
                # print("|> [sim_function] sim_function is feature_base")
                output, _ = model.m_step_forward(
                    batch,
                    None
                )
            elif sim_function == 'common_neighbor':
                # print("|> [sim_function] sim_function is common_neighbor")
                output, _ = model.m_step_forward(batch,
                                                 neighbors_info)
            elif sim_function == 'degree':
                # print("|> [sim_function] sim_function is common_neighbor")
                output, _ = model.m_step_forward(batch,
                                                 neighbors_info)
            output = output.float()
            loss_m = criterion(output, y)
            total_loss += loss_m.item()
            loss_m.backward()
            embedding_optimizer.step()

            pbar.update(batch.batch_size)

        avg_loss = total_loss / len(train_loader)
        #
        if avg_loss < least_loss:
            least_loss = avg_loss
            best_epoch = epoch + 1
            # best_valid_f1 = f1
            # best_model = copy.deepcopy(model)
            # torch.save(model.state_dict(), os.path.join(log_dir, "{}.pt".format(run_time)))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= args.patience:
            break

    print("Optimization Finished!")
    # used_time = time.time() - start_time

    torch.save(model.state_dict(), os.path.join(log_dir, "{}.pt".format(run_time)))

    test(
        model=model,
        test_dataset=computer_data,
        test_subgraph_loader=subgraph_loader,
        train_loader=train_loader,
        stats='Evaluating',
        device=device,
        run_time=run_time,
        log_dir=log_dir,
        args=args,
        enable_different_memorysize=enable_different_memorysize)


def test(model,
                           test_dataset,
                           test_subgraph_loader,
                           train_loader,
                           stats,
                           device,
                           run_time,
                           log_dir,
                           args,
                           enable_different_memorysize=False):
    model.eval()
    ys, preds = [], []
    y = test_dataset.y
    ys.append(y)

    gnn_start_time = timeit.default_timer()

    with torch.no_grad():
        training_embedding = []
        training_label = []
        training_id = []

        for batch in train_loader:
            batch = batch.to(device)
            batch_id = batch.n_id[:batch.batch_size]
            training_id.append(batch_id)
            output, embedding = model.m_step_forward(batch)
            training_embedding.append(embedding)
            y = batch.y[:batch.batch_size]
            training_label.append(y)

        training_embedding = torch.concat(training_embedding)
        training_label = torch.cat(training_label)
        training_id = torch.cat(training_id)

    # gnn_time = timeit.default_timer() - gnn_start_time
    # print("GNN time is ", gnn_time)
    test_batch_count = 0
    for _ in test_subgraph_loader:
        test_batch_count += 1
    # print("Test batch number", test_batch_count)

        dirichlet_macro_p, dirichlet_macro_r, dirichlet_macro_f1, dirichlet_micro_f1, dirichlet_num_of_nodes = memory_evaluate(
        model.embedding_encoder,
        model.lambda_,
        test_dataset,
        test_subgraph_loader,
        training_embedding,
        training_label,
        training_id,
        device)

    with open("{}/{}_memory_with_dirichlet_result.txt".format(log_dir,
                                                             str(run_time)),
              'w') as f:

        f.write(str(dirichlet_macro_p))
        f.write("\n")
        f.write(str(dirichlet_macro_r))
        f.write("\n")
        f.write(str(dirichlet_macro_f1))
        f.write("\n")
        f.write(str(dirichlet_micro_f1))
        f.write("\n")
        f.write(str(dirichlet_num_of_nodes))
        f.write("\n")

    original_macro_p, original_macro_r, original_macro_f1, original_micro_f1, original_num_of_nodes = orignal_evaluate(
        model.embedding_encoder,
        test_dataset,
        test_subgraph_loader,
        training_embedding,
        training_label)

    with open(
        "{}/{}_memory_with_original_result.txt".format(log_dir, str(run_time)),
        'w') as f:

        f.write(str(original_macro_p))
        f.write("\n")
        f.write(str(original_macro_r))
        f.write("\n")
        f.write(str(original_macro_f1))
        f.write("\n")
        f.write(str(original_micro_f1))
        f.write("\n")
        f.write(str(original_num_of_nodes))
        f.write("\n")

        print("Test set Dirichlet result:",
              "macro_p= {:.2f}".format(original_macro_p),
              "macro_r= {:.2f}".format(original_macro_r),
              "macro_f1= {:.2f}".format(original_macro_f1),
              "micro_f1= {:.2f}".format(original_micro_f1))


def memory_evaluate(_model,
                   _lambda_,
                   _dataset,
                   _subgraph_loader,
                   _training_embedding,
                   _training_label,
                   _training_id,
                   _device,
                   _threshold=5):

    _train_lambda = _lambda_[_training_id]
    prob = _train_lambda / _train_lambda.sum()
    res = []
    sorted_prob, indices = torch.sort(prob, descending=True)
    sum_ = 0.
    node_number_sum = 0

    for prob_i, id_i in zip(sorted_prob, indices):
        node_number_sum += 1
        sum_ += prob_i.item()
        res.append(id_i.item())
        if node_number_sum >= _threshold:
            break
        # if sum_ >= _threshold:
        #     break

    _optimized_training_embedding = torch.index_select(_training_embedding, 0, torch.tensor(res).to(_device))
    _optimized_training_label = torch.index_select(_training_label, 0, torch.tensor(res).to(_device))

    _model.eval()
    ys, preds = [], []
    y = _dataset.y
    ys.append(y)
    with torch.no_grad():
        output = _model.inference(_dataset.x,
                                         _subgraph_loader,
                                         _optimized_training_embedding,
                                         _optimized_training_label,
                                         "Evaluating")
        memory_size = _optimized_training_embedding.shape[0]
        predicts = output.max(1)[1].cpu()
        preds.append(predicts)
        gold, pred = torch.cat(ys, dim=0).cpu()[_dataset.test_mask], torch.cat(preds, dim=0).numpy()[_dataset.test_mask]
        macro_p = metrics.precision_score(gold, pred, average='macro') * 100
        macro_r = metrics.recall_score(gold, pred, average='macro') * 100
        macro_f1 = metrics.f1_score(gold, pred, average='macro') * 100
        micro_f1 = metrics.f1_score(gold, pred, average='micro') * 100
        return macro_p, macro_r, macro_f1, micro_f1, memory_size


def orignal_evaluate(_model,
                     _dataset,
                     _subgraph_loader,
                     _training_embedding,
                     _training_label):

    ys, preds = [], []
    y = _dataset.y
    ys.append(y)
    with torch.no_grad():
        output = _model.inference(_dataset.x,
                                        _subgraph_loader,
                                        _training_embedding,
                                        _training_label,
                                        "Evaluating")
        memory_size = _training_embedding.shape[0]
        predicts = output.max(1)[1].cpu()
        preds.append(predicts)
        gold, pred = torch.cat(ys, dim=0).cpu()[_dataset.test_mask], torch.cat(preds, dim=0).numpy()[_dataset.test_mask]
        acc = metrics.accuracy_score(gold, pred) * 100
        macro_p = metrics.precision_score(gold, pred, average='macro') * 100
        macro_r = metrics.recall_score(gold, pred, average='macro') * 100
        macro_f1 = metrics.f1_score(gold, pred, average='macro') * 100
        micro_f1 = metrics.f1_score(gold, pred, average='micro') * 100
        return macro_p, macro_r, macro_f1, micro_f1, memory_size


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=True, help='Use CUDA or not')
    parser.add_argument('--cuda_id', type=int, required=True,
                        help='Use CUDA or not')
    parser.add_argument('--epochs', type=int,
                        required=True, help='Number of epochs to train')
    parser.add_argument('--gnn_lr', type=float, default=0.005, help='Initial GNN learning rate')
    parser.add_argument('--vae_lr', type=float, default=0.005,
                        help='Initial VAE learning rate')
    parser.add_argument('--hidden_dim', type=int,
                        required=True,
                        help='Number of hidden units.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience')
    parser.add_argument('--normalize', required=True)
    parser.add_argument('--model', required=True,
                        help='training_model')
    parser.add_argument('--eta', type=float, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--k', type=int, required=True,
                        help='Number of similar nodes')
    parser.add_argument('--val_test_batch_size', type=int, required=True,
                        help='validation and test dataloader batch size')
    parser.add_argument('--criterion', type=str, required=True,
                        help='softmax or sigmoid')
    parser.add_argument('--run_times', type=int, required=True)
    parser.add_argument('--sim_function',
                        type=str,
                        required=True,
                        help='feature_base, common_neighbor')
    parser.add_argument('--enable_different_memorysize',
                        type=int,
                        default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args)
    run_times = args.run_times

    for run_time in range(run_times):
        print("|> Run time is ", run_time)
        train_test(run_time, args)
