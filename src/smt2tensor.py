from pysmt.operators import (FORALL, EXISTS, AND, OR, NOT, IMPLIES, IFF,
                             SYMBOL, FUNCTION,
                             REAL_CONSTANT, BOOL_CONSTANT, INT_CONSTANT,
                             PLUS, MINUS, TIMES, DIV,
                             LE, LT, EQUALS,
                             ITE)
import torch
import random
import math
import collections
import networkx as nx
from z3 import *

class Smtworkerror(RuntimeError):
    def __init__(self, arg):
        self.args = [arg]


class VariableInfo:
    def __init__(self, id, is_real, lower_bound=None, upper_bound=None):
        self.id = id
        self.is_real = is_real
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

class RealNodeInfo:
    def __init__(self, id, command, args):
        self.id = id
        self.command = command
        self.args = args
        self.task_graph = []

class myTensor(object):
    acc_eps = 0

    def __init__(self):
        self.nodes = []             # assert集合
        self.task_graph = []        # [[[command, nodeid, [argid_1, ...]], ...], ...]
        self.task_set = {}
        self.task_layer_cnt = 0
        self.tensor_args = {}
        self.and_task = []
        self.arg_cnt = -1
        self.names = []             # 变量名
        self.namemap = {}           # 变量名到数组位置映射(id, Real变量=True)

        self.__commands = {
            FORALL: self.__forall,
            EXISTS: self.__exists,
            AND: self.__and,
            OR: self.__or,
            NOT: self.__not,
            IMPLIES: self.__implies,
            IFF: self.__iff,
            PLUS: self.__plus,
            MINUS: self.__minus,
            TIMES: self.__times,
            DIV: self.__div,
            EQUALS: self.__equals,
            LE: self.__le,
            LT: self.__lt,
            ITE: self.__ite,
        }
        self.bool_commands = {
            self.__forall: FORALL,
            self.__exists: EXISTS,
            self.__and: AND,
            self.__or: OR,
            self.__not: NOT,
            self.__implies: IMPLIES,
            self.__iff: IFF,
        }
        self.bool_graph = {}    # {nodeid: [command, [argid_1, ...]}
        self.function_nodes = []
        self.real_nodes = {}    # {nodeid: [command, [argid_1, ...]}
        self.variable_dict = {}
        
    
    def new_node(self) -> int:
        self.arg_cnt += 1
        return self.arg_cnt


    def add_real_arg(self, name, lower_bound=None, upper_bound=None):
        id = self.new_node()
        self.namemap[name] = VariableInfo(id, True, lower_bound, upper_bound)
        self.names.append(name)

    def add_bool_arg(self, name):
        id = self.new_node()
        self.namemap[name] = VariableInfo(id, False)
        self.names.append(name)


    def parse_declare(self, symbol):
        type = symbol.symbol_type()
        name = symbol.symbol_name()
        if type.is_real_type():
            self.add_real_arg(name)
        elif type.is_bool_type():
            self.add_bool_arg(name)
        else:
            raise Smtworkerror("Undefined Type")

    def parse_assert(self, node):
        self.nodes.append(node)

    def make_not_node(self, node, layer, dim, is_and):
        node_id = self.new_node()
        if layer > self.task_layer_cnt:
            self.task_layer_cnt += 1
            self.task_graph.append([])
        tmp_list = []
        type = NOT
        tid = self.init_graph(node, layer+1, dim, 0)
        tmp_list.append(tid)
        tmp_set = self.task_set[tid]
        self.task_graph[layer].append(
            [self.__commands[type], node_id, tmp_list])
        self.task_set[node_id] = tmp_set
        if is_and and tmp_set:
            self.and_task.append(node_id)
        return node_id

    def init_graph(self, node, layer, dim, is_not, is_and=0):
        node_id = self.new_node()
        if node.is_not():               # 下传not
            is_not ^= 1
            return self.init_graph(node.arg(0), layer, dim, is_not, is_and)
        if is_not:
            if node.is_symbol() or node.is_constant() or node.is_equals():
                return self.make_not_node(node, layer, dim, is_and)

        if node.is_symbol():          # 符号
            return self.namemap[node.symbol_name()].id

        if layer > self.task_layer_cnt:
            self.task_layer_cnt += 1
            self.task_graph.append([])

        if node.is_constant():          # 常量
            x = node.constant_value()   # gmpy2类型
            self.task_graph[layer].append(
                [self.__constant, node_id, [float(x)]*dim])
            self.task_set[node_id] = set()
            return node_id

        tmp_list = []
        args = node.args()
        type = node.node_type()
        if is_not:          # 取反
            if type == AND:
                type = OR
                for arg in args:
                    tmp_list.append(self.init_graph(arg, layer+1, dim, 1))
            elif type == OR:
                type = AND
                for arg in args:
                    tmp_list.append(self.init_graph(
                        arg, layer+1, dim, 1, is_and))
            elif type == IMPLIES:
                type = AND
                tmp_list.append(self.init_graph(
                    args[0], layer+1, dim, 0, is_and))
                tmp_list.append(self.init_graph(
                    args[1], layer+1, dim, 1, is_and))
            elif type == IFF:
                tmp_list.append(self.init_graph(args[0], layer+1, dim, 0))
                tmp_list.append(self.init_graph(args[1], layer+1, dim, 1))
            elif type == LE:
                tmp_list.append(self.init_graph(args[1], layer+1, dim, 0))
                tmp_list.append(self.init_graph(args[0], layer+1, dim, 0))
            elif type == LT:
                tmp_list.append(self.init_graph(args[1], layer+1, dim, 0))
                tmp_list.append(self.init_graph(args[0], layer+1, dim, 0))
            else:
                raise Smtworkerror("Undefined")
        else:
            if type == AND:
                tis_and = is_and
            else:
                tis_and = 0
            for arg in args:
                tmp_list.append(self.init_graph(arg, layer+1, dim, 0, tis_and))
        self.task_graph[layer].append(
            [self.__commands[type], node_id, tmp_list])

        tmp_set = set()
        for tid in tmp_list:
            tmp_set = tmp_set | self.task_set[tid]
        self.task_set[node_id] = tmp_set
        if is_and and type != AND and tmp_set:
            self.and_task.append(node_id)

        # 变量上下界
        if is_and and type in (LE, LT, EQUALS):
            if args[0].is_symbol() and args[1].is_constant():
                var = self.namemap[args[0].symbol_name()]
                val = float(args[1].constant_value())
                if type in (LE, LT):
                    if var.upper_bound is None or var.upper_bound > val:
                        var.upper_bound = val
                else:
                    var.lower_bound = val
                    var.upper_bound = val
            elif args[1].is_symbol() and args[0].is_constant():
                var = self.namemap[args[1].symbol_name()]
                val = float(args[0].constant_value())
                if type in (LE, LT):
                    if var.lower_bound is None or var.lower_bound < val:
                        var.lower_bound = val
                else:
                    var.lower_bound = val
                    var.upper_bound = val

        return node_id

    def init_tensor(self, dim):
        for name in self.names:
            tmp_set = set()
            nid = self.namemap[name].id
            tmp_set.add(nid)
            self.task_set[nid] = tmp_set
        self.answer_id = self.new_node()
        self.task_graph.append([[self.__commands[AND], self.answer_id, []]])
        for node in self.nodes:
            self.task_graph[0][0][2].append(
                self.init_graph(node, 1, dim, 0, 1))

    def __forall(self, node):
        raise Smtworkerror("qwq")

    def __exists(self, node):
        raise Smtworkerror("qwq")

    def __constant(self, args):
        return torch.tensor(args, requires_grad=False)

    def __and(self, args):
        y = self.zeros
        for arg in args:
            if torch.all(self.tensor_args[arg] == self.falses):
                return self.falses
            if torch.all(self.tensor_args[arg] == self.trues):
                continue
            y = y + torch.max(self.zeros, self.tensor_args[arg])
        return y

    def __or(self, args):
        y = None
        for arg in args:
            if torch.all(self.tensor_args[arg] == self.trues):
                return self.trues
            if torch.all(self.tensor_args[arg] == self.falses):
                continue
            if y is not None:
                y = torch.min(y, self.tensor_args[arg])
            else:
                y = self.tensor_args[arg]
        if y is None:
            y = self.falses
        return y

    def __not(self, args):
        return -self.tensor_args[args[0]]

    def __implies(self, args):       # left -> right
        # return a<0?b:-a
        _a = -self.tensor_args[args[0]]
        _b = self.tensor_args[args[1]]
        return torch.where(_a < 0, _b, -_a)

    def __iff(self, args):           # left <-> right
        _a = self.tensor_args[args[0]]
        _b = self.tensor_args[args[1]]
        if torch.all(_a == self.trues):
            return _b
        elif torch.all(_a == self.falses):
            return -_b
        elif torch.all(_b == self.trues):
            return _a
        elif torch.all(_b == self.falses):
            return -_a
        return torch.where(_a > 0, -_b, _b)+torch.where(_b > 0, -_a, _a)

    def __plus(self, args):
        y = self.zeros
        for arg in args:
            y = y + self.tensor_args[arg]
        return y

    def __minus(self, args):
        return self.tensor_args[args[0]] - self.tensor_args[args[1]]

    def __times(self, args):
        y = self.ones
        for arg in args:
            y = y * self.tensor_args[arg]
        return y

    def __div(self, args):
        return self.tensor_args[args[0]] / self.tensor_args[args[1]]

    def __equals(self, args):
        y = (self.tensor_args[args[0]]-self.tensor_args[args[1]])
        return y*y + torch.log(1 + y*y)
        # return y*y

    def __le(self, args):
        y = torch.max(self.zeros, self.tensor_args[args[0]] - self.tensor_args[args[1]])
        return y*y + torch.log(1 + y)
        # return self.tensor_args[args[0]] - self.tensor_args[args[1]]

    def __lt(self, args):
        y = torch.max(self.zeros, self.tensor_args[args[0]] - self.tensor_args[args[1]])
        return y*y + torch.log(1 + y)
        # return self.tensor_args[args[0]] - self.tensor_args[args[1]]

    def __ite(self, args):       # if( iff ) then  left  else  right
        raise Smtworkerror("qwq")

    def init_val(self, dim=1):
        self.zeros = torch.zeros(dim, dtype=torch.float)
        self.ones = torch.ones(dim, dtype=torch.float)
        self.trues = torch.full((dim,), float('-inf'))
        self.falses = torch.full((dim,), float('inf'))
        tmp_list = []
        for name in self.names:
            nid = self.namemap[name].id
            if self.namemap[name].is_real:   # REAL
                val = [0.15 - random.random() * 0.1 for _ in range(dim)]
            else:
                val = [1 - random.randint(0, 1) * 2 for _ in range(dim)]
            tmp_list.append(val)
        self.vars = torch.tensor(tmp_list, requires_grad=True)
        l = len(tmp_list)
        for i in range(l):
            self.tensor_args[i] = self.vars[i]

    
    def init_bool_tree(self):
        for layer in self.task_graph:
            for oper in layer:
                fun = oper[0]
                if fun in (self.__and, self.__or, self.__not, self.__implies, self.__iff): # bool
                    fun = self.bool_commands[fun]
                    self.bool_graph[oper[1]] = [fun, oper[2]]
                elif fun in (self.__equals, self.__le, self.__lt):
                    self.real_nodes[oper[1]] = RealNodeInfo(oper[1], fun, oper[2])
                    self.function_nodes.append(oper[1])
                else:
                    self.real_nodes[oper[1]] = RealNodeInfo(oper[1], fun, oper[2])
    
    def build_task_graph(self, node_id):
        fun = self.real_nodes[node_id].command
        args = self.real_nodes[node_id].args
        task_graph = [[[fun, node_id, args]]]
        for nid in args:
            if nid not in self.real_nodes:
                continue
            tg = [list(layer) for layer in self.real_nodes[nid].task_graph]
            # 合并task_graph
            depth = len(task_graph)
            for i in range(len(tg)):
                if i+1 < depth:
                    task_graph[i+1].extend(tg[i])
                else:
                    task_graph.append(tg[i])
        self.real_nodes[node_id].task_graph = [list(layer) for layer in task_graph]

    def init_real_tree(self):
        for layer in reversed(self.task_graph):
            for oper in layer:
                node_id = oper[1]
                if node_id not in self.real_nodes:
                    continue
                self.build_task_graph(node_id)

    
    def build_expression(self, node_id):
        if node_id in self.function_nodes:
            if node_id in self.variable_dict:
                return self.variable_dict[node_id]
            else:
                var = Bool('xn' + str(node_id))
                self.variable_dict[node_id] = var
                return var
        fun, args = self.bool_graph[node_id]
        if fun == AND:
            return And(*[self.build_expression(arg) for arg in args])
        elif fun == OR:
            return Or(*[self.build_expression(arg) for arg in args])
        elif fun == NOT:
            return Not(self.build_expression(args[0]))
        elif fun == IMPLIES:
            return Implies(self.build_expression(args[0]), self.build_expression(args[1]))
        elif fun == IFF:
            return And(Implies(self.build_expression(args[0]), self.build_expression(args[1])), 
                                Implies(self.build_expression(args[1]), self.build_expression(args[0])))

    def build_goal(self):
        goals = []
        for node_id in self.function_nodes:
            if self.real_nodes[node_id].command == self.__equals:
                goals.append(If(self.variable_dict[node_id], 100, 0))
            else:
                goals.append(If(self.variable_dict[node_id], 1, 0))
        return Sum(goals)
    
    def bool_sol(self):
        expr = self.build_expression(self.answer_id)
        opt = Optimize()
        opt.add(expr)
        optimization_expr = self.build_goal()
        opt.minimize(optimization_expr)
        if opt.check() != sat:
            return None
        model = opt.model()
        
        need_vars = []
        for nid in self.function_nodes:
            val = model.eval(self.variable_dict[nid])
            if val == True:
                need_vars.append(nid)

        return need_vars
    
    def new_sol(self):
        self.init_bool_tree()
        self.init_real_tree()
        need_vars = self.bool_sol()
        if need_vars is None:
            return None
        
        task_graph = [[[self.__commands[AND], self.answer_id, need_vars]]]
        for nid in need_vars:
            depth = len(task_graph)
            tg = [list(layer) for layer in self.real_nodes[nid].task_graph]
            for i in range(len(tg)):
                if i+1 < depth:
                    task_graph[i+1].extend(tg[i])
                else:
                    task_graph.append(tg[i])

        self.real_nodes[self.answer_id] = RealNodeInfo(self.answer_id, self.__and, need_vars)
        self.real_nodes[self.answer_id].task_graph = [list(layer) for layer in task_graph]


    def get_need_vars(self):
        return self.real_nodes[self.answer_id].args
    
    def new_iteration(self, nid = None):
        if nid is None:
            nid = self.answer_id
        task_graph = self.real_nodes[nid].task_graph
        for layer in reversed(task_graph):
            for oper in layer:
                fun = oper[0]
                self.tensor_args[oper[1]] = fun(oper[2])

        return self.tensor_args[nid]

    # 检查如果变量超过边界，则强制赋值
    def bound_check(self):
        for name in self.names:
            v = self.namemap[name]
            lb = v.lower_bound
            ub = v.upper_bound
            with torch.no_grad():
                if lb is not None:
                    self.vars[v.id] = torch.max(self.vars[v.id], torch.tensor(lb))
                if ub is not None:
                    self.vars[v.id] = torch.min(self.vars[v.id], torch.tensor(ub))

    # def pre_sol(self):                      # 化简常量
    #     for layer in reversed(self.task_graph):
    #         for oper in layer:
    #             fun = oper[0]
    #             ts = fun(oper[2])
    #             if fun not in (self.__equals, self.__le, self.__lt) or not torch.allclose(ts, ts[0]):
    #                 self.tensor_args[oper[1]] = ts
    #                 continue
    #             # 都是一个值的话基本说明变量改变无影响，因此说明这里是常量
    #             if fun == self.__equals:
    #                 self.tensor_args[oper[1]] = self.trues if ts[0] == 0.0 else self.falses
    #             elif fun == self.__le:
    #                 self.tensor_args[oper[1]] = self.trues if ts[0] <= 0.0 else self.falses
    #             elif fun == self.__lt:
    #                 self.tensor_args[oper[1]] = self.trues if ts[0] < 0.0 else self.falses

    #     return self.tensor_args[self.answer_id]

    def sol(self):
        for layer in reversed(self.task_graph):
            for oper in layer:
                fun = oper[0]
                self.tensor_args[oper[1]] = fun(oper[2])

        return self.tensor_args[self.answer_id]

    def run(self, initval):
        self.zeros = torch.zeros(1, dtype=torch.float)
        self.ones = torch.ones(1, dtype=torch.float)
        self.trues = torch.full((1,), float('-inf'))
        self.falses = torch.full((1,), float('inf'))
        grads = {}
        for (key, val, grad) in initval:
            nid = self.namemap[key].id
            if self.namemap[key].is_real:        # Real
                val = float(format(val, '.2g'))
                if abs(val) < 1e-5:
                    val = 0
            grads[nid] = grad
            self.tensor_args[nid] = torch.tensor([val])
        self.sol()

        subsets = [self.task_set[tid]
                   for tid in self.and_task if self.tensor_args[tid][0] > 0]
        if len(subsets) == 0:
            return [(key, float(val)) for (key, val, grad) in initval]
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
            matching = nx.algorithms.bipartite.maximum_matching(B, top_nodes=funcs)
            result = [matching[subset] for subset in funcs if subset in matching]
        except:
            result = []
        return [(key, float(val)) for (key, val, grad) in initval if self.namemap[key].id not in result]

    def print_args(self, mss=None):
        if mss:
            print(mss)
        l = len(self.names)
        for i in range(l):
            print(self.names[i], ":", self.vars[i].item())
