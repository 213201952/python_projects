import copy
import numpy
import random
graph = [] #定义耦合图的集合
quantum_circuit = [] #定义逻辑电路
result_circuit = [] #定义实际电路
qubits = [0, 1, 2, 3, 4, 5] #定义逻辑比特映射,qubits每一位的位置代表一个逻辑比特，每一位的值代表映射到物理比特的位置。例如qubits[1]=0表示逻辑比特1映射到了物理比特0
positions = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] #定义耦合图上的点的映射,未被映射的点被标记为-1
distance_table = numpy.zeros((16, 16), int)
layers = []  # 定义函数的层



def QX5(): #生成QX5的耦合图
    graph.clear()
    graph.append([1, 0])
    graph.append([1, 2])
    graph.append([2, 3])
    graph.append([3, 14])
    graph.append([3, 4])
    graph.append([5, 4])
    graph.append([6, 5])
    graph.append([6, 11])
    graph.append([6, 7])
    graph.append([7, 10])
    graph.append([8, 7])
    graph.append([9, 8])
    graph.append([9, 10])
    graph.append([11, 10])
    graph.append([12, 5])
    graph.append([12, 11])
    graph.append([12, 13])
    graph.append([13, 4])
    graph.append([13, 14])
    graph.append([15, 0])
    graph.append([15, 14])
    graph.append([15, 2])
    # print('QPU耦合图为:', graph)

def circuit_generation(num_gate,num_qubit):
    for i in range(num_gate):
        gate = random.choices(['RX','RY','RZ','CX','iSWAP'],[1,1,1,3,3])
        if gate[0] == 'RX' or gate[0] == 'RY' or gate[0] == 'RZ':
            gate.append(random.randint(0, num_qubit-1))
            quantum_circuit.append(gate)
        else:
            while 1:
                a = random.randint(0, num_qubit-1)
                b = random.randint(0, num_qubit-1)
                if(a != b):
                    break
            gate.append(a)
            gate.append(b)
            quantum_circuit.append(gate)
    print(quantum_circuit)

def greed_search_length(start, end): #找到最短路径
    queue = [] #存储所有可能的路径
    route = [] #当前路径
    route.append(start)
    queue.append(route)
    length = 0 #路径长度
    next_nodes = [] #路径下一步拓展的点的集合

    def contains(list, node):
        for n in list:
            if n == node:
                return False
        return True

    while queue:
        route = queue[0]
        del queue[0]
        current = route[-1] #-1表示最末尾
        if current == end: #表示已经找到终点，结束循环
            length = len(route)
            break
        else:
            next_nodes.clear()
            for edge in graph:
                if edge[0] == current and contains(route, edge[1]):
                    next_nodes.append(edge[1])
                if edge[1] == current and contains(route, edge[0]):
                    next_nodes.append(edge[0])

            for next_node in next_nodes:
                #print('本轮拓展节点为', next_node)
                route1 = copy.deepcopy(route) #深拷贝
                #print('此时route1为',route1)
                route1.append(next_node)
                #print('此时route1为',route1)
                queue.append(route1)
                #print('此时queue为', queue)
    return length

def bulid_distance_table():
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                distance_table[i][j] = greed_search_length(i, j)
    print(distance_table) #输出两点距离矩阵

def greed_search_route(start, end): #找到最短路径
    solutions = []  # 定义为点与点之间最短路径的集合
    queue = [] #存储所有可能的路径
    route = [] #当前路径
    route.append(start)
    queue.append(route)
    length = 0 #路径长度
    next_nodes = [] #路径下一步拓展的点的集合
    solutions.clear() #每次寻找前清空上一次的结果

    while queue:
        route = queue[0]
        del queue[0]
        current = route[-1]
        if current == end: #表示已经找到终点，结束循环
            length = len(route)
            solutions.append(route)
            #print('跳出循环时solutions为', solutions)
            break
        else:
            next_nodes.clear()
            for edge in graph:
                if edge[0] == current and contains(route, edge[1]):
                    next_nodes.append(edge[1])
                if edge[1] == current and contains(route, edge[0]):
                    next_nodes.append(edge[0])

            for next_node in next_nodes:
                #print('本轮拓展节点为', next_node)
                route1 = copy.deepcopy(route) #深拷贝
                #print('此时route1为',route1)
                route1.append(next_node)
                #print('此时route1为',route1)
                queue.append(route1)
                #print('此时queue为', queue)

    while queue and len(queue[0]) == length:
        if queue[0][-1] == end:
            solutions.append(queue[0])
            #print('此时solutions为', solutions)
        del queue[0]
    # print('最终最短路径为', solutions)
    return solutions

def move_to_edge(start1, start2, end1, end2):

    def greed_search_route1(start, end, obstacle):  # 找到最短路径
        solutions = []  # 定义为点与点之间最短路径的集合
        queue = []  # 存储所有可能的路径
        route = []  # 当前路径
        route.append(start)
        queue.append(route)
        length = 0  # 路径长度
        next_nodes = []  # 路径下一步拓展的点的集合
        solutions.clear()  # 每次寻找前清空上一次的结果

        def contains(list, node):
            for n in list:
                if n == node:
                    return False
            return True

        while queue:
            route = queue[0]
            del queue[0]
            current = route[-1]
            if current == end:  # 表示已经找到终点，结束循环
                length = len(route)
                solutions.append(route)
                # print('跳出循环时solutions为', solutions)
                break
            else:
                next_nodes.clear()
                for edge in graph:
                    if edge[0] == current and contains(route, edge[1]) and contains(route, obstacle):
                        next_nodes.append(edge[1])
                    if edge[1] == current and contains(route, edge[0]) and contains(route, obstacle):
                        next_nodes.append(edge[0])

                for next_node in next_nodes:
                    # print('本轮拓展节点为', next_node)
                    route1 = copy.deepcopy(route)  # 深拷贝
                    # print('此时route1为',route1)
                    route1.append(next_node)
                    # print('此时route1为',route1)
                    queue.append(route1)
                    # print('此时queue为', queue)

        while queue and len(queue[0]) == length:
            if queue[0][-1] == end:
                solutions.append(queue[0])
                # print('此时solutions为', solutions)
            del queue[0]
        # print('最终最短路径为', solutions)
        return solutions

    if distance_table [start1][end1] + distance_table [start2][end2] > distance_table [start1][end2] + distance_table [start2][end1]:
        start1, start2 = start2, start1 #因为是无向图，选择最短路径进行映射
    print('start1=',start1,'start2=',start2)
    result = []
    result1 = greed_search_route1(start1, end1, start2)
    result.append(result1[0])
    result2 = greed_search_route1(start2, end2, start1)
    result.append(result2[0])
    print(result)
    return result

def mapping(): #将逻辑比特映射到物理电路
    for i in range(len(qubits)): #遍历所有逻辑比特的映射
        qubits[i] = i #仅实现简单映射
        positions[qubits[i]] = i
    # print('positions为',positions)

def swap(node1, node2): #node1、node2表示拓扑图上需要交换的两个物理比特所代表的逻辑比特映射
    positions[node1],positions[node2] = positions[node2],positions[node1] #交换物理比特的映射
    qubits[positions[node1]],qubits[positions[node2]] = positions.index(positions[node1]),positions.index(positions[node2]) #交换逻辑比特的映射
    result_circuit.append(['swap', node1, node2])
    # print('交换后positions为', positions)
    # print('交换后qubits为', qubits)

def routing(tuple): #添加交换门
    gate = tuple
    done = 0
    for edge in graph:
        if (gate[1] == edge[0] and gate[2] == edge[1]) or (gate[1] == edge[1] and gate[2] == edge[0]):
            result_circuit.append(gate)
            done = 1
            # print('此时电路图1为:', classical_circuit)
            break
    if done == 0: #如果不能直接执行，则添加交换门
        start = gate[1]
        end = gate[2]
        greed_search_route(start, end)
        change = solutions[0]
        for i in range(len(change) - 2):  # 进行交换操作
            swap(change[i], change[i+1])
        result_circuit.append(gate)
        # print('此时电路图2为:', classical_circuit)

def layering():
    layer = [] #每一层具体包含的门
    label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    label_gate = []

    def search(node): #找到了输出Ture
        for i in label_gate:
            if node == i:
                return True
        return False

    while len(label_gate) != len(quantum_circuit):
        for g in range(len(quantum_circuit)): #找到所有可以并行执行的门
            if search(g):
                continue
            gate = quantum_circuit[g]
            if (label[gate[1]] == 0 and label[gate[2]] == 0): #如果这个门满足并行执行的条件
                layer.append(gate) #将他添加到层中
                label[gate[1]], label[gate[2]] = 1,1 #将他作用的比特标记上
                label_gate.append(g) #从逻辑电路中标记这个门，表示已经完成分层

                for j in range(g, len(quantum_circuit)):  # 找到余下门中通过互易规则合并到同一层的门
                    if search(j):
                        continue
                    jgate = quantum_circuit[j]
                    if (jgate[1] == gate[1] and jgate[2] == gate[2]) or (
                            jgate[1] == gate[2] and jgate[2] == gate[1]):  # 判断这个门的作用的比特是否相同，接下再判断能否互易
                        m1, n1 = 0, 0  # 记录这两个比特中间夹了有几个门
                        for k in range(g + 1, j):  # 如果作用的比特相同，则遍历这两个作用比特相同的门之间的门，判断这些门是否会影响互易
                            if search(k):
                                continue
                            kgate = quantum_circuit[k]
                            if gate[1] == kgate[1] or gate[1] == kgate[2]:
                                m1 = m1 + 1
                            if gate[2] == kgate[1] or gate[2] == gate[2]:
                                n1 = n1 + 1
                        if m1 == 0 or n1 == 0:  # 通过夹的门个数判断能否互易，能互易则添加到同一层中
                            layer.append(jgate)
                            label_gate.append(j)

            else:
                label[gate[1]], label[gate[2]] = 1, 1  # 将他作用的比特标记上



        layers.append(layer)
        print('layer=',layer)
        layer.clear()
        label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return layers

QX5() #生成QX5耦合图
mapping() #将逻辑比特映射到物理电路
bulid_distance_table()
