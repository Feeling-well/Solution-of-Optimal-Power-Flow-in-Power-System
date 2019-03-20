# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:33:58 2017
lastest edit：2017-12-1
功能：opf潮流程序
@author: 苏
"""

###将模型转化为二阶锥，参照文献Strong SOCP Relaxations for the Optimal Power Flow
##心得体会，python适配gurobi的的确确是比matlab好很多，速度也快。
##虽然是转化成socp，但是在python的gurobi平台上直接套用qcp模型就可以进行求解。
##在功率平衡的等式约束中，由于没有设定j不等于i，所以和参考文献上有所不同，但实际上是一样的。
##gurobi不能设置等式约束，但是参照文献所说，可以近似不等式约束，最后结果也还算理想。

import numpy as np
from scipy import sparse
import time
from gurobipy import *
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
#import math
#import scipy.io as sio
#import pprint
#np.set_printoptions(threshold=np.inf)     #输出全部矩阵数据。
#np.set_printoptions(threshold = 1e6)      #输出精度与全部数据控制语句
#from numpy import *
#import pandas as pd
#import time

import cvxopt
import cvxopt.solvers
from cvxopt import matrix, solvers

#%% 符号说明
# node_number：节点数量;    Jacobi：雅可比矩阵;       blanNode：平衡节点号;
# data：系统参数;          lineblock：线路参数;      branchblock：接地支路参数;
# transblock：变压器参数;   pvNodepara：pv节点参数
# y：节点导纳矩阵           pvNode：PV节点号;         pvNum：PV节点数量
# pis：节点注入有功;        qis：节点注入无功
# iteration：迭代次数;      accuracy：迭代精度;      jie：修正量
# deltP：有功不平衡量       deltQ：无功不平衡量       delta：相角修正量    deltv：电压修正量
# va:相角                   v0：幅值
#%% 数据读取


data=[]
for l in open('D:\\pythonfiles\\118.txt','r'):
    row = [float(x) for x in l.split()]
    row+=[0]*(8-len(row))
    data.append(row)
data_np=np.array(data)
start = time.time()     #开始计时
node_number = int(data_np[0][0])    #节点数量
data[2][1]                          #平衡节点号
myreturn_index = []
for i in np.arange(len(data)):
    if data[i][0] == 0:
        myreturn_index.append(i)
myreturn = np.array(myreturn_index)
lineN=myreturn[1]-myreturn[0]-1     #线路参数的行数
knum=myreturn[1]-1                  #线路参数结束行数
#读取线路参数
lineblock=data_np[myreturn[0]+1:knum+1]   #线路参数整体切片
lineNo=lineblock[0:lineN,0]
linei=lineblock[0:lineN,1]          #线路参数的母线i
linej=lineblock[0:lineN,2]          #线路参数的母线j
liner=lineblock[0:lineN,3]          #线路参数的R
linex=lineblock[0:lineN,4]          #线路参数的X
lineb=lineblock[0:lineN,5]          #线路参数的B
#接地支路参数读取
branch=myreturn[2]-myreturn[1]-1    #接地支路共有行数
k1=knum+2                           #接地支路开始行
k2=knum+1+branch                    #接地支路结束行
branchblock=data_np[k1:k2+1,:]      #接地支路参数整体切片
branchi=branchblock[0:branch,0]     #接地支路节点号
branchb=branchblock[0:branch,1]     #接地支路导纳
branchg=branchblock[0:branch,2]
#变压器参数读取
trans=myreturn[3]-myreturn[2]-1     #变压器参数共有行数
k1=k2+2                             #变压器参数开始行
k2=myreturn[2]+trans                #变压器参数结束行
transblock=data_np[k1:k2+1,:]       #变压器参数整块切片
transi=transblock[0:trans,1]        #变压器参数的母线i
transj=transblock[0:trans,2]        #变压器参数的母线j
transr=transblock[0:trans,3]        #变压器参数的R
transx=transblock[0:trans,4]        #变压器参数的X
transk=transblock[0:trans,5]        #变压器参数的变比
#节点功率参数读取
pow=myreturn[4]-myreturn[3]-1       #节点功率共有行数
k1=k2+2                             #节点功率开始行
k2=k2+1+pow
powblock=data_np[k1:k2+1,:]        #节点功率参数整块切片
powi=powblock[0:pow,0]             #节点功率参数的节点号
powpgi=powblock[0:pow,1]/100           #节点功率参数的PG
powqgj=powblock[0:pow,2]/100           #节点功率参数的QG
powpdi=powblock[0:pow,3]/100          #节点功率参数的PD
powqdj=powblock[0:pow,4]/100          #节点功率参数的QD


pv=myreturn[5]-myreturn[4]-1       #PV节点共有行数
k1=k2+2                            #PV节点开始行
k2=k2+1+pv;                        #PV节点结束行

diats = [str(x) for x in np.arange(node_number)+1]  #字符表示节点标号‘1’-‘1047’
#读取pv节点参数
pvblock=data_np[k1:k2+1,:]
generate_reac_diat = [str(int(x)) for x in pvblock[:,0]]
non_generate_reac_diat = list(set(diats)-set(generate_reac_diat))
pvi=pvblock[0:pv,0]                #PV节点参数的节点号
pvv=pvblock[0:pv,1]                #PV节点参数的电压
pvqmin=pvblock[0:pv,2]/100            #PV节点参数的Qmin
pvqmax=pvblock[0:pv,3]/100            #PV节点参数的Qmax

pvv_diat = dict(zip(generate_reac_diat,(pvv).tolist()))
pvqmax = dict(zip(generate_reac_diat,(pvqmax).tolist()))
pvqmin = dict(zip(generate_reac_diat,(pvqmin).tolist()))



#发电机参数读取
Gfactor=myreturn[6]-myreturn[5]-1    #接地支路共有行数

k1=k2+2                           #接地支路开始行
k2=k2+1+Gfactor                    #接地支路结束行
Gfactorblock=data_np[k1:k2+1,:]      #接地支路参数整体切片
generate_ac_diat = [str(int(x)) for x in Gfactorblock[:,0]]
non_generate_ac_diat = list(set(diats)-set(generate_ac_diat))

Gc = Gfactorblock[0:Gfactor,1]       #发电机参数c
Gb = Gfactorblock[0:Gfactor,2]       #发电机参数b
Ga = Gfactorblock[0:Gfactor,3]       #发电机参数a

Ga_diat = dict(zip(generate_ac_diat,Ga))
Gb_diat = dict(zip(generate_ac_diat,Gb))
Gc_diat = dict(zip(generate_ac_diat,Gc))

Gupper = Gfactorblock[0:Gfactor,5]/100   #发电机上限
Glower = Gfactorblock[0:Gfactor,4]/100   #发电机下限

Gupper = dict(zip(generate_ac_diat,Gupper))
Glower = dict(zip(generate_ac_diat,Glower))



#%% 数据读取完毕，导纳矩阵的形成
#符合说明：linei为行，linej为列
#线路导纳矩阵
z1=1.*(liner+1j*linex)**-1                                       #矩阵的除法
z11=1.*(liner+1j*linex)**-1+1j*lineb
y1_1=-sparse.coo_matrix((z1,(linei-1,linej-1)),shape=(node_number,node_number))
y1_2=-sparse.coo_matrix((z1,(linej-1,linei-1)),shape=(node_number,node_number))
y1_3=sparse.coo_matrix((z11,(linei-1,linei-1)),shape=(node_number,node_number))
y1_4=sparse.coo_matrix((z11,(linej-1,linej-1)),shape=(node_number,node_number))
y1=y1_1+y1_2+y1_3+y1_4                                           #线路导纳矩阵
#变压器导纳矩阵
z2=1*(transr+1j*transx)**-1*(transk)**-1                         #  含义为1./(transr+j*transx)./transk
z22=(1-transk)*(transr+1j*transx)**-1*(transk)**-1*(transk)**-1+z2
z23=(transk-1)*(transr+1j*transx)**-1*(transk)**-1+z2
y2_1=-sparse.coo_matrix((z2,(transi-1,transj-1)),shape=(node_number,node_number))
y2_2=-sparse.coo_matrix((z2,(transj-1,transi-1)),shape=(node_number,node_number))
y2_3=sparse.coo_matrix((z22,(transi-1,transi-1)),shape=(node_number,node_number))
y2_4=sparse.coo_matrix((z23,(transj-1,transj-1)),shape=(node_number,node_number))
y2=y2_1+y2_2+y2_3+y2_4                                           #变压器导纳矩阵
#接地支路导纳矩阵
y3=sparse.coo_matrix((branchg+1j*branchb,(branchi-1,branchi-1)),shape=(node_number,node_number))
y=y1+y2+y3                                                        #节点导纳矩阵

class Example(dict):
    def __getitem__(self,item):
        try:
            return dict.__getitem__(self,item)
        except KeyError:
            value = self[item] = type(self)()
            return value
y_g = Example()    #实例复合字典
y_b = Example()    #实例复合字典


#np.real(y)[101].indices+1该数组的表示101+1节点对应的自导纳点和互导纳点，第一个数为自导纳，剩余为互导纳


for i in range(node_number):
    for j in range(len(np.real(y)[i].indices)-1):
        y_g[i+1][(np.real(y)[i].indices+1)[j+1]] = -np.real(y)[i].data[j]
    y_g[i + 1][i+1] = np.real(y)[i].data[0]

for i in range(node_number):
    for j in range(len(np.imag(y)[i].indices)-1):
        y_b[i+1][(np.imag(y)[i].indices+1)[j+1]] = -np.imag(y)[i].data[j]
    y_b[i + 1][i+1] = np.imag(y)[i].data[0]


def strfloat(diat):
    diat = float(diat)
    return (diat)

def strint(diat):
    diat = int(diat)
    return (diat)

def floatstr(diat):
    diat = str(diat)
    return (diat)

def fintstr(diat):
    diat = int(diat)
    diat = str (diat)
    return (diat)

gurobi_martix_dict_g = { (diat_1, diat_2) : np.real(y[strint(diat_1)-1,strint(diat_2)-1]) for diat_1 in diats for diat_2 in diats }
gurobi_martix_dict_b = { (diat_1, diat_2) : np.imag(y[strint(diat_1)-1,strint(diat_2)-1]) for diat_1 in diats for diat_2 in diats }


# gurobi_martix_dict_g = { (diat_1, diat_2) : node_martix_dict_g[diat_1][diat_2] for diat_1 in diats for diat_2 in diats }
# gurobi_martix_dict_b = { (diat_1, diat_2) : node_martix_dict_b[diat_1][diat_2] for diat_1 in diats for diat_2 in diats }

y_abs=abs(y)
pis_1=(powpgi-powpdi)/100                                         #基准值处理
pis_2=(powqgj-powqdj)/100                                         #基准值处理
powi0=powi*0
powi0[0:node_number-1]=0
pis=sparse.coo_matrix((pis_1,(powi-1,powi0)))/100                #pis与qis的求解结果正确
qis=sparse.coo_matrix((pis_2,(powi-1,powi0)))/100                #
powi0=np.transpose(powi0)
v0=(node_number,1)                                               #单位矩阵
v0=np.ones(v0)                                                   #初始化电压值
va=powi0*0                                                       #初始化电压相角
v0[int(data[2][1])-1]=1                                          #电压初始化
n=0
#为pv节点电压赋值
for i in range(0,len(pvi)):
    v0[int(pvi[i])-1]=pvv[n]
    n=n+1
accuracy=1  #精度
#电压上下限约束
voltageupper = 1.1*np.ones((node_number))
voltagelower = 0.9*np.ones((node_number))

## gurobi建模求解
#
# m = Model('sxy_OPF')
#
# active_power_diat = {}
# reactive_power_diat = {}
# voltage_real_diat = {}
# voltage_imag_diat = {}
#
# # for i in diats:
# #     voltage_real_diat[i] = v0[strint(i)-1]
# #     voltage_imag_diat[i] = 0
#
# for i in diats:
#     active_power_diat[i] = m.addVar(lb = 0,name = 'active_power_diat')
#     reactive_power_diat[i] = m.addVar(lb = -99.9,name = 'reactive_power_diat')
#     voltage_real_diat[i] = m.addVar(lb = -99.9,name = 'voltage_real_diat')
#     voltage_imag_diat[i] = m.addVar(lb = -99.9, name = 'voltage_imag_diat')
#
# C = {}
# S = {}
#
# for i in diats:
#     for j in diats:
#         C[i, j] = m.addVar(lb = -99.9 ,name = 'C')
#         S[i, j] = m.addVar (lb = -99.9  ,name='S')
#
#
# #非发电机节点的有功出力置0
# m.addConstrs((active_power_diat[diat] == 0 for diat in non_generate_ac_diat),name="balancePVP")
# #非发电机节点的无功出力置0
# m.addConstrs((reactive_power_diat[diat] == 0 for diat in non_generate_reac_diat),name="balancePVQ")
#
#
# #有功约束
# m.addConstrs((active_power_diat[diat] <= Gupper[diat] for diat in generate_ac_diat),name="active_Capacity_upper")
# m.addConstrs((active_power_diat[diat] >= Glower[diat] for diat in generate_ac_diat),name="active_Capacity_lower")
#
# #无功约束
# m.addConstrs((reactive_power_diat[diat] <= pvqmax[diat] for diat in generate_reac_diat),name="reactive_Capacity_upper")
# m.addConstrs((reactive_power_diat[diat] >= pvqmin[diat] for diat in generate_reac_diat),name="reactive_Capacity_lower")
#
# #PV节点电压约束
# m.addConstrs((voltage_real_diat[diat] == pvv_diat[diat] for diat in generate_reac_diat),name='PV')
#
# #系统负荷
# active_load_diat = dict(zip(diats,powpdi.tolist()))
# reactive_load_diat = dict(zip(diats,powqdj.tolist()))
#
# #gurobi_martix_dict_b[diat_i,diat_j]*voltage_real_diat[diat_j]) for diat_j in diats) for diat_i in diats
# #voltage_real_diat['1'] * (y_g[strint (1)][strint (1)] * voltage_real_diat['1'] - y_b[strint (1)][strint (1)] *voltage_imag_diat['1'])
# #voltage_imag_diat['1'] * (y_g[strint (1)][strint (1)] * voltage_imag_diat['1'] + y_b[strint (1)][strint (1)] * voltage_real_diat['1'])
#
# # m.addConstrs (((gurobi_martix_dict_g[diat_i, diat_i] * C[diat_i, diat_i] + \
# #                         quicksum(gurobi_martix_dict_g[diat_i, fintstr(diat_j)] * C[diat_i, fintstr(diat_j)] \
# #                  - gurobi_martix_dict_b[diat_i, fintstr(diat_j)] * S[diat_i, fintstr(diat_j)] \
# #                  for diat_j in y_g[strint (diat_i)] )) \
# #                == active_power_diat[diat_i] - active_load_diat[diat_i] for diat_i in diats), name="balanceP")
#
# # for i in diats:
# #     for j in diats:
# #         if not gurobi_martix_dict_b[i,j]:
# #             gurobi_martix_dict_b[i,j] = 0.0
# #             gurobi_martix_dict_g[i, j] = 0.0
#
# def constrainp(i):
#     a = gurobi_martix_dict_b[i, i] * S[i, i]
#     for j in diats:
#         a += gurobi_martix_dict_g[i,j]*C[i,j]-gurobi_martix_dict_b[i,j]*S[i,j]
#     return (a)
#
# def constrainq(i):
#     a = -gurobi_martix_dict_g[i, i] * S[i, i]
#     for j in diats:
#         a += -gurobi_martix_dict_b[i,j]*C[i,j]-gurobi_martix_dict_g[i,j]*S[i,j]
#     return (a)
#
# m.addConstrs((constrainp(i)==active_power_diat[i] - active_load_diat[i]) for i in diats)
# m.addConstrs((constrainq(i)==reactive_power_diat[i] - reactive_load_diat[i]) for i in diats)
#
# m.addConstr(C[fintstr(data[2][1]),fintstr(data[2][1])]==1)
# m.addConstrs((C[diat_i, diat_i]<=1.1*1.1 for diat_i in diats),name='ClimUP')
# m.addConstrs((C[diat_i, diat_i]>=0.1*0.1 for diat_i in diats),name='ClimLow')
# m.addConstrs((C[diat_i, diat_j]==C[diat_j, diat_i] for diat_j in diats  for diat_i in diats  ),name='C_Constrs')
# m.addConstrs((S[diat_i, diat_j]==-S[diat_j, diat_i] for diat_j in diats  for diat_i in diats),name='S_Constrs')
#
# for diat_i in diats:
#     for diat_j in diats:
#         m.addQConstr((C[diat_i, diat_j]*C[diat_i, diat_j]+S[diat_i, diat_j]*S[diat_i, diat_j])<=C[diat_i, diat_i]*C[diat_j, diat_j])
#
# obj = quicksum(Ga_diat[diat] * active_power_diat[diat]* active_power_diat[diat] + Gb_diat[diat] * active_power_diat[diat]
#                          + Gc_diat[diat] for diat in generate_ac_diat)
#
# m.setObjective(obj)
# m.optimize()



