import sys
import time
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from tqdm import *
from scipy.sparse import *

import GNNAdvisor as GNNA           # import GNNAdvisor
import pubmed_util
from gnn_conv import *
from dataset import *
import datetime

#torch.set_printoptions(edgeitems=3703/2)
parser = argparse.ArgumentParser()
# Dataset related parameters.
parser.add_argument("--dataDir", type=str, default="../osdi-ae-graphs", help="the path to graphs")
parser.add_argument("--dataset", type=str, default='reddit', help="dataset")
parser.add_argument("--dim", type=int, default=32, help="input embedding dimension size")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension size")# this number will not affacted the result
parser.add_argument("--classes", type=int, default=22, help="output classes size")

# Model training related parameters.
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin'],  help="GCN or GIN")
parser.add_argument("--num_epoches", type=int, default=200, help="number of epoches for training, default=200")

# Manually set the performance related parameters
parser.add_argument("--partSize", type=int, default=32, help="neighbor-group size")
parser.add_argument("--dimWorker", type=int, default=32, help="number of worker threads (MUST < 32)")
parser.add_argument("--warpPerBlock", type=int, default=4, help="number of warp per block, recommended: GCN: 8, GIN: 2")
parser.add_argument("--sharedMem", type=int, default=100, help="shared memory size of each block (Quadro P6000 64(KB) sm_61), default=100(KB) for RTX3090 sm_86")

# Additional flags for studies.
parser.add_argument('--manual_mode', type=str, choices=['True', 'False'], default='True', help="True: use manual config, False: auto config, default: True")
parser.add_argument('--verbose_mode', type=str, choices=['True', 'False'], default='False', help="True: verbose mode, False: simple mode, default: False")
parser.add_argument('--enable_rabbit', type=str, choices=['True', 'False'], default='False', help="True: enable rabbit reordering, False, disable rabbit reordering, default: False (disable for both manual and auto mode).")
parser.add_argument('--loadFromTxt', type=str, choices=['True', 'False'], default='True', help="True: load the graph TXT edge list, False: load from .npy, default: False (load from npz fast)")
parser.add_argument('--single_spmm', type=str, choices=['True', 'False'], default='False', help="True: profile the single SpMM (neighbor aggregation) kernel for number epoches times")
parser.add_argument('--verify_spmm', type=str, choices=['True', 'False'], default='False', help="True: verify the output correctness of a single SpMM (neighbor aggregation) kernel against the CPU reference implementation.")

args = parser.parse_args()
print(args)

partSize, dimWorker, warpPerBlock, sharedMem = args.partSize, args.dimWorker, args.warpPerBlock, args.sharedMem
manual_mode = args.manual_mode == 'True'
verbose_mode = args.verbose_mode == 'True'
enable_rabbit = args.enable_rabbit == 'True'
loadFromTxt = args.loadFromTxt == 'True'
single_spmm = args.single_spmm == 'True'
verify_spmm = args.verify_spmm == 'True'
num_test_feature = int(args.dim)
print("88888888 num_of _test_feat:", num_test_feature)
# requires GPU for evaluation.
assert torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################################
# loading data from files
####################################
if loadFromTxt:
    #path = osp.join(args.dataDir, args.dataset)
    print("load from txt")

    #dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=True, verbose=verbose_mode)
    #path = "/mnt/huge_26TB/data/test2/reddit/graph_structure/reddit_graph_undirect.txt"
    #dim = 602
    #classes = 41
    #path = "/home/ygong07/data/test2/citeseer/graph_structure/citeseer_graph_undirect.txt"
    #path = "/home/ygong07/data/test2/citeseer/graph_structure"
    """
    if args.dataset == 'citeseer':
        path = "/home/ygong07/data/test2/citeseer/graph_structure/citeseer_graph_undirect.txt"
        dim = 3703
        classes = 6
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/home/ygong07/data/test2/citeseer/feature/citeseer_feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/home/ygong07/data/test2/citeseer/index/citeseer_train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/home/ygong07/data/test2/citeseer/label/citeseer_y_label.txt")
    elif args.dataset == 'cora':
        path = "/home/ygong07/data/test2/cora/graph_structure/cora.txt"
        dim = 1433
        classes = 7
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/home/ygong07/data/test2/cora/feature/cora_feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/home/ygong07/data/test2/cora/index/cora_train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/home/ygong07/data/test2/cora/label/cora_y_label.txt")

    elif args.dataset == 'pubmed':
        path = "/home/ygong07/data/test2/pubmed/graph_structure/pubmed_graph_undouble.txt"
        dim = 500
        classes = 3
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/home/ygong07/data/test2/pubmed/feature/feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/home/ygong07/data/test2/pubmed/index/train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/home/ygong07/data/test2/pubmed/label/y_label.txt")

    elif args.dataset == 'ogb-arx':
        #path = "/home/ygong07/data/test2/ogb-arx/graph_structure/graph_undirect.txt"
        path = "/home/ygong07/data/test2/ogb-arx/graph_structure_double/ogb_arx_double.txt"
        dim = 128
        classes = 40
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/home/ygong07/data/test2/ogb-arx/feature/feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/home/ygong07/data/test2/ogb-arx/index/train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/home/ygong07/data/test2/ogb-arx/label/train_y_label.txt")

    elif args.dataset == 'reddit':
        #path = "/mnt/huge_26TB/data/test2/reddit/graph_structure/reddit_graph_undirect.txt"
        path = "/mnt/huge_26TB/data/test2/reddit/graph_structure_double/reddit_double.txt"
        dim = 602
        classes = 40
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        #feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/reddit/feature/reddit_feature.txt")
        num_vcount = 232965
        feat = torch.ones(num_vcount, num_test_feature)
        #feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/reddit/index/reddit_train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/reddit/label/reddit_y_label.txt")
    elif args.dataset == 'ogb-product':
        path = "/home/ygong07/data/test2/ogb-product/graph_structure_double/ogb_product_double.txt"
        #path = "/home/ygong07/data/test2/ogb-product/graph_structure/graph_undirect.txt"
        dim = 100
        classes = 47
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/home/ygong07/data/test2/ogb-product/feature/feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/home/ygong07/data/test2/ogb-product/index/train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/home/ygong07/data/test2/ogb-product/label/train_y_label.txt")
"""



    #train_id = torch.tensor(train_id)
    #train_y_label = torch.tensor(train_y_label)
    #train_id=train_id.to(device)
    #train_y_label=train_y_label.to(device)

"""
else:
    path = osp.join(args.dataDir, args.dataset+".npz")
    dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=False, verbose=verbose_mode)
    feat = pubmed_util.read_feature_info("/home/ygong07/data/test2/citeseer/feature/citeseer_feature.txt")
    feat = torch.tensor(feat)
    feat = feat.to(device)
"""   

#path = "/home/ygong07/data/test2/ogb-product/graph_structure/graph_undirect.txt"
dataset = custom_dataset(args.dataset, args.dim, args.classes, load_from_txt=True, verbose=verbose_mode)
feat = torch.ones(dataset.num_nodes, args.dim)
feat = feat.to(device)
#print("feat from loadtxt", feat)


num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
print("graph node and edge", num_nodes, num_edges)
column_index = dataset.column_index
row_pointers = dataset.row_pointers
degrees = dataset.degrees

####################################
# Building input property profile.
####################################
inputInfo = inputProperty(row_pointers, column_index, degrees, 
                            partSize, dimWorker, warpPerBlock, sharedMem,
                            hiddenDim=args.hidden, dataset_obj=dataset, enable_rabbit=enable_rabbit,
                            manual_mode=manual_mode, verbose=verbose_mode)

####################################
# Decider for parameter selection.
####################################
inputInfo.decider()

inputInfo = inputInfo.set_input()
if verbose_mode:
    print('----------------------------')
    inputInfo.print_param()
    print()

inputInfo = inputInfo.set_hidden()
if verbose_mode:
    inputInfo.print_param()
    print()
    print('----------------------------')
# sys.exit(0)

####################################
# Building neighbor partitioning.
####################################
start = time.perf_counter()
partPtr, part2Node = GNNA.build_part(inputInfo.partSize, inputInfo.row_pointers)
build_neighbor_parts = time.perf_counter() - start
if verbose_mode:
    print("# Build nb_part (s): {:.3f}".format(build_neighbor_parts))

inputInfo.row_pointers  = inputInfo.row_pointers.to(device)
inputInfo.column_index  = inputInfo.column_index.to(device)
inputInfo.partPtr = partPtr.int().to(device)
inputInfo.part2Node  = part2Node.int().to(device)



################# test spmm for reddit################
######################################################
def test_spmm(X, inputInfo):
    weights = torch.nn.Parameter(torch.randn(200, 16)).to(device) # this weight will not be used in following computation
    #ctx.inputInfo = inputInfo
    #ctx.partSize, ctx.dimWorker, ctx.warpPerBlock = \
                    #inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock

    start_train = time.perf_counter()
    X_prime = GNNA.forward(X, weights, inputInfo.row_pointers, inputInfo.column_index, 
                                  inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node, \
                                                                  inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock)[0]
    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_train
    print("feat dim", num_test_feature)
    print('Time (second for spmm): {:.6f}'.format(train_time))
    print("\n")
    print("\n")



dataset  = dataset.to(device)



if __name__ == '__main__':
    # dry run
    #for _ in range(2):
        #train()
    # exit(0)

    torch.cuda.synchronize()
    test_spmm(feat, inputInfo.set_input())





