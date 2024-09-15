from pysmt.smtlib.parser import SmtLibParser, get_formula
import pysmt.smtlib.commands as smtcmd
import argparse
import time
import torch
from smt2tensor import myTensor
from tqdm import tqdm
import re
import os
import numpy as np
from sklearn.cluster import KMeans
import numpy as np

DEBUG = False
MAXWORKERS = None       # ProcessPoolExecutor default is None
DIM = 1024
Z3TIMELIMIT = 10000     # ms
THREADTIMELIMIT = Z3TIMELIMIT/1000
PROCESSTIMELIMIT = THREADTIMELIMIT + 1
ITERTIMELIMIT = 600     # s
Epochs = 600
Lr = 0.5
USEEPS = False
EPS = 0.0001


def parse_args():
    arg_parser = argparse.ArgumentParser(description='SMT_GD')
    arg_parser.add_argument('path', type=str, help='path of smt-lib file')
    arg_parser.add_argument("-D", "--debug", action="store_true", help="enable debug mode")
    arg_parser.add_argument("-W", "--workers", type=int, default=None, help="number of parallel workers")
    args_get = arg_parser.parse_args()
    return args_get


mypath=""

def get_smt_script(path):
    smt_parser = SmtLibParser()
    global mypath
    mypath = path
    with open(path, 'r') as fp:
        script_get = smt_parser.get_script(fp)
        smt_logic = smt_parser.logic
    return script_get, smt_logic


def get_smt_formula(path):
    with open(path, 'r') as fp:
        f = get_formula(fp)
    return f

parameter_val={}
def generate_parameter(declaration):
    outstr = ""
    data = parameter_val[declaration.strip()]
    for (a,b,c) in data:
        outstr += f" {a} {b} {c}"
    return outstr

def z3sol_protect(namemap, init_result, subsets):
    if len(subsets) > 0:
        vars = set()
        for subset in subsets:
            vars |= subset
        funcs = [(i,) for i in range(len(subsets))]
        vars = list(vars)
        B = nx.Graph()          # 二分图最大匹配
        B.add_nodes_from(funcs, bipartite=0)
        B.add_nodes_from(vars, bipartite=1)
        for i, subset in enumerate(subsets):
            for var in subset:
                B.add_edge((i,), var)
        try:
            matching = nx.algorithms.bipartite.maximum_matching(
                B, top_nodes=funcs)
            result = [matching[subset]
                      for subset in funcs if subset in matching]
        except:
            result = []
        # init_result = [(key, val) for (key, val)
        #                in init_result if namemap[key][0] not in result]
        for (key, val) in init_result:
            number = val

            formatted_number = float(f"{number:.2g}")
            str_number = str(formatted_number)
            decimal_index = str_number.find('.') + 1
            first_nonzero_index = next((i for i, digit in enumerate(str_number[decimal_index:]) if digit != '0'), None)

            # 计算误差范围
            if first_nonzero_index is not None:
                precision = 10 ** -(first_nonzero_index + 2)
            else:
                precision = 0.01

            formatted_number = float(f"{number:.2g}")
            if abs(formatted_number) < 1e-5:
                formatted_number = 0
                precision = 1e-5
            lower_bound = formatted_number - precision
            upper_bound = formatted_number + precision
            min_ = format(float(format(lower_bound, '.2g')),'f')
            max_ = format(float(format(upper_bound, '.2g')),'f')

            if key not in parameter_val:
                parameter_val[key] = []
            if namemap[key][0] not in result:
                parameter_val[key].append((min_, max_, 1))
            else:
                parameter_val[key].append((min_, max_, 0))



def parallel_sol(init_result, mytensor, out_time):
    res = False
    init_vals = [[] for _ in range(DIM)]
    for key_, (vals_, grads_) in init_result.items():
        values = []
        for i in range(DIM):
            val = vals_[i]
            if mytensor.namemap[key_][1]:       # Real
                val = float(format(val, '.2g'))
                if abs(val) < 1e-5:
                    val = 0
            init_vals[i].append((key_, val))
            values.append(val)
        nid = mytensor.namemap[key_][0]
        mytensor.tensor_args[nid] = torch.tensor(values)
    mytensor.sol()

    for i in range(DIM):
        subsets = [mytensor.task_set[tid]
                    for tid in mytensor.and_task if mytensor.tensor_args[tid][i] > 0]
        z3sol_protect(mytensor.namemap, init_vals[i], subsets)

    
    with open(mypath, 'r') as file:
        smt2_content = file.read()
    smt2_content = re.sub(r'(declare-fun (.+?)\(\) Real)', lambda m: f'{m.group(0)} GD {DIM}{generate_parameter(m.group(2))}', smt2_content)
    smt2_content = re.sub(r'(declare-const (.+?) Real)', lambda m: f'{m.group(0)} GD {DIM}{generate_parameter(m.group(2))}', smt2_content)
    # print(smt2_content)

    file_path = str(out_time)+"/"+ mypath
    folder_path = os.path.dirname(file_path)
    # print(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path, 'w') as file:
        file.write(smt2_content)



def MyPrint(init_result, cluster_stats, N, outtime):
    global parameter_val
    id_ = 0
    for key_, (vals_, grads_) in init_result.items():
        data = []
        for i in range(N):
            min_ = format(cluster_stats[i][0][id_], 'f')
            max_ = format(cluster_stats[i][1][id_], 'f')
            cnt_ = str(cluster_stats[i][2])
            data.append((min_, max_, cnt_))
        parameter_val[key_] = data
        id_+=1

    with open(mypath, 'r') as file:
        smt2_content = file.read()
    smt2_content = re.sub(r'(declare-fun (.+?)\(\) Real)', lambda m: f'{m.group(0)} GD {N}{generate_parameter(m.group(2))}', smt2_content)
    smt2_content = re.sub(r'(declare-const (.+?) Real)', lambda m: f'{m.group(0)} GD {N}{generate_parameter(m.group(2))}', smt2_content)
    # print(smt2_content)

    file_path = str(outtime)+"/"+ mypath
    folder_path = os.path.dirname(file_path)
    # print(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path, 'w') as file:
        file.write(smt2_content)


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if (epoch % 50 == 0) and (epoch > 0):
        lr = lr * (0.5 ** (epoch // 50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def init_tensor(script):
    mytensor = myTensor()
    for cmd in script:
        if cmd.name in [smtcmd.DECLARE_FUN, smtcmd.DECLARE_CONST]:
            symbol = cmd.args[0]
            mytensor.parse_declare(symbol)
        elif cmd.name == smtcmd.ASSERT:
            node = cmd.args[0]
            mytensor.parse_assert(node)
    mytensor.init_tensor(DIM)
    return mytensor


def clustering(data, num_clusters):
    data_array = np.array([[vals_[i] for key_, (vals_, grads_) in data.items()] for i in range(DIM)])
    # print(data_array)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data_array)
    labels = kmeans.labels_
    cluster_stats = []
    for i in range(num_clusters):
        cluster_data = data_array[labels == i]  # 获取属于当前簇的数据
        cluster_min = np.min(cluster_data, axis=0)  # 取值下界
        cluster_max = np.max(cluster_data, axis=0)  # 取值上界
        cluster_count = len(cluster_data)  # 数据数量
        cluster_stats.append((cluster_min, cluster_max, cluster_count))
    
    cluster_stats = sorted(cluster_stats, key=lambda x: -x[2])
    return cluster_stats



def generate_init_solution(mytensor):
    global Epochs, Lr
    init_result = {}
    mytensor.init_val(DIM)
    T1 = time.process_time()
    y = mytensor.pre_sol()  # 化简常量，顺便记录sol时间
    T2 = time.process_time()
    if T2-T1 > 0.8:
        new_epochs = int(ITERTIMELIMIT*0.5/(T2-T1))
        Lr *= new_epochs/Epochs
        Epochs = new_epochs

    if DEBUG:
        print('程序运行时间1:%s毫秒' % ((T2 - T1)*1000))
        print(Epochs)

    optimizer = torch.optim.Adam([mytensor.vars], lr=Lr)

    _out_cnt = 100###
    for step in tqdm(range(Epochs)) if DEBUG else range(Epochs):
        adjust_learning_rate(optimizer, step, Lr)
        optimizer.zero_grad()
        y = mytensor.sol()

        y.backward(torch.ones(DIM))
        for name in mytensor.names:
            init_result[name] = (mytensor.vars[mytensor.namemap[name][0]].tolist(),
                                 mytensor.vars.grad[mytensor.namemap[name][0]])

        T2 = time.process_time()

        if torch.any(y < torch.zeros(DIM)):
            break
        if T2-T1 > ITERTIMELIMIT:
            break
        optimizer.step()

    
    parallel_sol(init_result, mytensor, _out_cnt)


    return init_result




def solve(path):
    script, smt_logic = get_smt_script(path)
    formula = get_smt_formula(path)

    mytensor = init_tensor(script)
    init_result = generate_init_solution(mytensor)

    # if init_result is None:         # sat
    #     print("sat")
    # else:
    #     res = parallel_sol(init_result, mytensor, formula, smt_logic)
    #     print("sat" if res else "NONE")


if __name__ == "__main__":
    T1 = time.perf_counter()
    args = parse_args()
    DEBUG = args.debug
    MAXWORKERS = args.workers
    
    solve(args.path)
    if DEBUG:
        print(time.perf_counter()-T1)
