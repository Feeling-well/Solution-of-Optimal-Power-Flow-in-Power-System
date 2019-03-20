## gurobi建模求解
from powerflow import *

m = Model('sxy_OPF')

active_power_diat = {}
reactive_power_diat = {}
voltage_real_diat = {}
voltage_imag_diat = {}


for i in diats:
    active_power_diat[i] = m.addVar(lb = 0,name = 'active_power_diat')
    reactive_power_diat[i] = m.addVar(lb = -99.9,name = 'reactive_power_diat')
    voltage_real_diat[i] = m.addVar(lb = -99.9,name = 'voltage_real_diat')
    voltage_imag_diat[i] = m.addVar(lb = -99.9, name = 'voltage_imag_diat')

C = {}
S = {}

for i in diats:
    for j in diats:
        C[i, j] = m.addVar(lb = -99.9 ,name = 'C')
        S[i, j] = m.addVar (lb = -99.9  ,name='S')


#非发电机节点的有功出力置0
m.addConstrs((active_power_diat[diat] == 0 for diat in non_generate_ac_diat),name="balancePVP")
#非发电机节点的无功出力置0
m.addConstrs((reactive_power_diat[diat] == 0 for diat in non_generate_reac_diat),name="balancePVQ")


#有功约束
m.addConstrs((active_power_diat[diat] <= Gupper[diat] for diat in generate_ac_diat),name="active_Capacity_upper")
m.addConstrs((active_power_diat[diat] >= Glower[diat] for diat in generate_ac_diat),name="active_Capacity_lower")

#无功约束
m.addConstrs((reactive_power_diat[diat] <= pvqmax[diat] for diat in generate_reac_diat),name="reactive_Capacity_upper")
m.addConstrs((reactive_power_diat[diat] >= pvqmin[diat] for diat in generate_reac_diat),name="reactive_Capacity_lower")

#PV节点电压约束
m.addConstrs((voltage_real_diat[diat] == pvv_diat[diat] for diat in generate_reac_diat),name='PV')

#系统负荷
active_load_diat = dict(zip(diats,powpdi.tolist()))
reactive_load_diat = dict(zip(diats,powqdj.tolist()))


def constrainp(i):
    a = gurobi_martix_dict_b[i, i] * S[i, i]
    for j in diats:
        a += gurobi_martix_dict_g[i,j]*C[i,j]-gurobi_martix_dict_b[i,j]*S[i,j]
    return (a)

def constrainq(i):
    a = -gurobi_martix_dict_g[i, i] * S[i, i]
    for j in diats:
        a += -gurobi_martix_dict_b[i,j]*C[i,j]-gurobi_martix_dict_g[i,j]*S[i,j]
    return (a)

m.addConstrs((constrainp(i)==active_power_diat[i] - active_load_diat[i]) for i in diats)
m.addConstrs((constrainq(i)==reactive_power_diat[i] - reactive_load_diat[i]) for i in diats)

m.addConstr(C[fintstr(data[2][1]),fintstr(data[2][1])]==1)
m.addConstrs((C[diat_i, diat_i]<=1.1*1.1 for diat_i in diats),name='ClimUP')
m.addConstrs((C[diat_i, diat_i]>=0.1*0.1 for diat_i in diats),name='ClimLow')
m.addConstrs((C[diat_i, diat_j]==C[diat_j, diat_i] for diat_j in diats  for diat_i in diats  ),name='C_Constrs')
m.addConstrs((S[diat_i, diat_j]==-S[diat_j, diat_i] for diat_j in diats  for diat_i in diats),name='S_Constrs')



for diat_i in diats:
    for diat_j in diats:
        m.addQConstr((C[diat_i, diat_j]*C[diat_i, diat_j]+S[diat_i, diat_j]*S[diat_i, diat_j])<=C[diat_i, diat_i]*C[diat_j, diat_j])

obj = quicksum(Ga_diat[diat] * active_power_diat[diat]* active_power_diat[diat] + Gb_diat[diat] * active_power_diat[diat]
                         + Gc_diat[diat] for diat in generate_ac_diat)

m.setObjective(obj)
m.optimize()
