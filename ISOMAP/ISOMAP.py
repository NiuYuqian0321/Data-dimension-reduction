#coding=utf-8
import numpy  as np
from scipy.sparse import csgraph
import operator
import time

def load_dataset(file_name, delimiter='\t'):
	dataArr = []
	labelArr = []
	with open(file_name) as f:
		for line in f:
			stringLine = line.strip().split(delimiter)
			dataLine = list(map(float, stringLine))
			labelArr.append(dataLine.pop())
			dataArr.append(dataLine)
	return np.mat(dataArr), labelArr

def dijkstra(data_matrix, start_node):
    '''
    Dijkstra求解最短路径算法
    输入：原始数据矩阵，起始顶点
    输出；起始顶点到其他顶点的最短距离
    '''

    data_matrix = np.array(data_matrix)
    vex_num=len(data_matrix)
    flag_list=['False']*vex_num
    dist=['0']*vex_num

    for i in range(vex_num):
        flag_list[i]=False
        dist[i]=data_matrix[start_node][i]

    flag_list[start_node]=False
    dist[start_node]=0
    
    k=0
    for i in range(1, vex_num):
        min_value = float("inf")

        for j in range(vex_num):
            if flag_list[j]==False and dist[j]!=float("inf") and dist[j]<min_value:
                min_value=dist[j]
                k=j


        flag_list[k]=True
        for j in range(vex_num):
            if data_matrix[k][j]==float("inf") or flag_list[j] == True:
                continue
            else:    
                dist[j] = min(dist[j],min_value+data_matrix[k][j])

    return dist
    

def isomap(dataset,K):
	#mat转换为array
	dataset = np.array(dataset)
	#计算每个样本到其他样本的欧氏距离，构成矩阵G
	numSamples = dataset.shape[0]
	G = np.mat(np.zeros((numSamples,numSamples)))
	index = 0
	for data in dataset:
		diff = np.tile(data, (numSamples, 1)) - dataset
		squaredDiff = diff ** 2
		squaredDist = squaredDiff.sum(axis = 1)
		distance = squaredDist ** 0.5
		#对每一行计算出来的距离进行从小到大排序
		sortedDistIndices = np.argsort(distance)
		#最小的K个距离保留，其余设置无穷大
		for i in range(len(distance)):
			if distance[i] > distance[sortedDistIndices[K]]:
				distance[i] = float("inf")
		G[index] = distance
		index = index + 1

	#计算样本间的最短路径dijkstra, 构成距离矩阵D
	#该方法正确，调用了包scipy
	#start = time.time()
	D1 = csgraph.shortest_path(G, 'D')
	#print(type(D1))
	#end = time.time()
	#print(end-start)	

	#该方法错误，因为返回的D2类型为matrix，D**2计算的结果不是元素的平方，而是矩阵的平方
	#D2 = np.mat(np.zeros((numSamples,numSamples)))
	#for i in range(numSamples):
	#	D2[i] = dijkstra(G, i)
	#	print i ,"/",numSamples,"dijkstra"
	#print(type(D2))
	
	return D1

