from gurobipy import *
from data_inputs import *
from sim_data import *
import settings

# param_dict = settings.param_dict

##### STOCHASTIC #####

def optimize_vpp_stochastic(param_dict):
    p,w,l = get_input_data(param_dict)
    p_sim, w_sim, l_sim, sim_perm = get_sim_data(param_dict)

    model = Model("VPP")
    model.ModelSense = -1  # set to maximize

    b,g,s,v,f = {},{},{},{},{}

    for t in range(settings.T):
        b[t] = model.addVar(lb=-1*param_dict['B_MAX'], ub=param_dict['B_MAX'], vtype="C", name="b(%s)"%(t))  # battery flow
        g[t] = model.addVar(vtype="B", name="g(%s)"%(t))  # gas turbine running?
        s[t] = model.addVar(vtype="B", name="s(%s)"%(t))  # gas turbine started?
        v[t] = model.addVar(lb=0, ub=param_dict['V_MAX'], vtype="C", name="m(%s)"%(t))  # battery storage

    revenue = model.addVars(len(sim_perm), obj = 1.0/len(sim_perm), vtype="C", lb=-10e10, name='revenue')

    model.update()

    for idx, sim in enumerate(sim_perm):
        p = p_sim['sim_%d'%sim[0]]
        w = w_sim['sim_%d'%sim[1]]
        l = l_sim['sim_%d'%sim[2]]

        f[idx] = model.addVars(settings.T, vtype="B", name='fine')

        for t in range(settings.T):
            # model.addConstr((b[t] + p[t] + w[t] + g[t]*settings.G) >= settings.MIN_GEN_LIST[t], "Min_Generation(%s)"%t)

            model.addConstr((param_dict['min_gen_list'][t] - (b[t] + p[t] + w[t] + param_dict['turb_cap']*g[t])) <= settings.M*f[idx][t], 'Fine_1')
            model.addConstr((param_dict['min_gen_list'][t] - (b[t] + p[t] + w[t] + param_dict['turb_cap']*g[t])) >= -1*settings.M*(1-f[idx][t]), 'Fine_2')

        model.addConstr((revenue[idx] == quicksum(l[t]*(b[t] + p[t] + w[t] + settings.G*g[t]) - (param_dict['CG']*g[t])\
                                - param_dict['S']*s[t] - f[idx][t]*param_dict['fine'] for t in range(settings.T))), name='set_revenue')

    for t in range(settings.T):
        model.addConstr(b[t] <= param_dict['B_MAX'], "Max Batt Charge(%s)"%t)
        model.addConstr(b[t] >= -1*param_dict['B_MAX'], "Max Batt Discharge(%s)"%t)
        model.addConstr(v[t] <= param_dict['V_MAX'], "Max Storage(%s)"%t)
        model.addConstr(v[t] >= 0, "Min Storage(%s)"%t)

    for t in range(1,settings.T):
        model.addConstr(g[t] - g[t-1] <= s[t], "Startup Criteria(%s)"%t)
        model.addConstr(v[t] == (v[t-1] + settings.B_EFF*b[t]), "Batt Storage(%s)"%t)

    # set initial conditions for r[0] and v[0]
    model.addConstr(g[0] == 0, "Turbine Init Cond")
    model.addConstr(v[0] == 0.5*param_dict['V_MAX'], "Batt Storage Init Cond")
    model.update()
    ## write out to file
    # model.write('VPP_stochastic.lp')

    # Run
    model.Params.OutputFlag = 0 # 0 = silent mode
    model.optimize()

    return model,b,g,s,v,f,revenue


def get_stochastic_vpp_output(param_dict):
    p,w,l = get_input_data(param_dict)
    model,b,g,s,v,f,revenue = optimize_vpp_stochastic(param_dict)

    df = pd.DataFrame(list(range(settings.T)), columns=['hour'])
    df['batt_flow'] = [round(b[t].X,2) for t in range(settings.T)]
    df['batt_energy'] = [round(v[t].X,2) for t in range(settings.T)]
    df['turb_on'] = [int(g[t].X) for t in range(settings.T)]
    df['turb_start'] = [int(s[t].X) for t in range(settings.T)]
    df['turb_gen'] = df['turb_on'] * param_dict['turb_cap']
    df['solar_gen'] = [round(p[t],2) for t in range(settings.T)]
    df['wind_gen'] = [round(w[t],2) for t in range(settings.T)]
    df['net_gen'] = df['solar_gen'] + df['turb_gen'] + df['wind_gen']
    df['net_out'] = df['net_gen'] + df['batt_flow']
    df['elect_price'] = [round(l[t],2) for t in range(settings.T)]
    df['hourly_profit'] = df['elect_price']*df['net_out'] - param_dict['CG']*df['turb_on']\
                    - param_dict['S']*df['turb_start']
    df['fine'] = [0 if (df['net_out'][t] >= param_dict['min_gen_list'][t])\
                    else param_dict['fine'] for t in range(settings.T)]

    rev = []
    fine = []
    for sim in range(param_dict['sims']**3):
        rev.append(round(revenue[sim].X,2))
        fine.append(sum(f[sim][t].X for t in range(settings.T)))

    return df,rev,fine


##### STATIC #####

def optimize_vpp(param_dict):
    p,w,l = get_input_data(param_dict)
    model = Model("VPP")

    b,g,s,v,f = {},{},{},{},{}

    for t in range(settings.T):
        b[t] = model.addVar(lb=-1*param_dict['B_MAX'], ub=param_dict['B_MAX'], vtype="C", name="b(%s)"%(t))  # battery flow
        g[t] = model.addVar(vtype="B", name="g(%s)"%(t))  # gas turbine running?
        s[t] = model.addVar(vtype="B", name="s(%s)"%(t))  # gas turbine started?
        v[t] = model.addVar(lb=0, ub=param_dict['V_MAX'], vtype="C", name="m(%s)"%(t))  # battery storage
        f[t] = model.addVar(vtype="B", name='fine')  #fine
    model.update()

    # Set objective
    obj = LinExpr()
    obj = quicksum(l[t]*(b[t] + p[t] + w[t] + param_dict['turb_cap']*g[t]) - (param_dict['CG']*g[t])\
                                - param_dict['S']*s[t] - f[t]*param_dict['fine'] for t in range(settings.T))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    for t in range(settings.T):
        model.addConstr(b[t] <= param_dict['B_MAX'], "Max Batt Charge(%s)"%t)
        model.addConstr(b[t] >= -1*param_dict['B_MAX'], "Max Batt Discharge(%s)"%t)
        model.addConstr(v[t] <= param_dict['V_MAX'], "Max Storage(%s)"%t)
        model.addConstr(v[t] >= 0, "Min Storage(%s)"%t)
        # model.addConstr((b[t] + p[t] + w[t] + g[t]*settings.G) >= settings.MIN_GEN_LIST[t], "Min_Generation(%s)"%t)
        model.addConstr((param_dict['min_gen_list'][t] - (b[t] + p[t] + w[t] + param_dict['turb_cap']*g[t]))\
                            <= settings.M*f[t], 'Fine_1')
        model.addConstr((param_dict['min_gen_list'][t] - (b[t] + p[t] + w[t] + param_dict['turb_cap']*g[t]))\
                            >= -1*settings.M*(1-f[t]), 'Fine_2')


    for t in range(1,settings.T):
        model.addConstr(g[t] - g[t-1] <= s[t], "Startup Criteria(%s)"%t)
        model.addConstr(v[t] == (v[t-1] + settings.B_EFF*b[t]), "Batt Storage(%s)"%t)


    # set initial conditions for r[0] and v[0]
    model.addConstr(g[0] == 0, "Turbine Init Cond")
    model.addConstr(v[0] == 0.5*param_dict['V_MAX'], "Batt Storage Init Cond")
    model.update()

    ## write out to file
    # model.write('VPP_stochastic.lp')

    # Run
    model.Params.OutputFlag = 0 # 0 = silent mode
    model.optimize()

    return model,b,g,s,f,v

def get_static_vpp_output(param_dict):
    p,w,l = get_input_data(param_dict)
    model,b,g,s,f,v = optimize_vpp(param_dict)

    df = pd.DataFrame(list(range(settings.T)), columns=['hour'])
    df['batt_flow'] = [round(b[t].X,2) for t in range(settings.T)]
    df['batt_energy'] = [round(v[t].X,2) for t in range(settings.T)]
    df['turb_on'] = [int(g[t].X) for t in range(settings.T)]
    df['turb_start'] = [int(s[t].X) for t in range(settings.T)]
    df['turb_gen'] = (df['turb_on'] * settings.G).round(2)
    df['solar_gen'] = [round(p[t],2) for t in range(settings.T)]
    df['wind_gen'] = [round(w[t],2) for t in range(settings.T)]
    df['net_gen'] = (df['solar_gen'] + df['turb_gen'] + df['wind_gen']).round(2)
    df['net_out'] = (df['net_gen'] + df['batt_flow']).round(2)
    df['elect_price'] = [round(l[t],2) for t in range(settings.T)]
    df['hourly_profit'] = (df['elect_price']*df['net_out'] - param_dict['CG']*df['turb_on']\
                    - param_dict['S']*df['turb_start']).round(2)
    df['fine'] = [round(f[t].X*param_dict['fine'],2) for t in range(settings.T)]
    return df


##### STOCHASTIC, NO BATTERY #####

def optimize_vpp_stochastic_no_battery(param_dict):
    p,w,l = get_input_data(param_dict)
    p_sim, w_sim, l_sim, sim_perm = get_sim_data(param_dict)
    model = Model("VPP")
    model.ModelSense = -1  # set to maximize

    g,s,f = {},{},{}
    for t in range(settings.T):
        g[t] = model.addVar(vtype="B", name="g(%s)"%(t))  # gas turbine running?
        s[t] = model.addVar(vtype="B", name="s(%s)"%(t))  # gas turbine started?
    revenue = model.addVars(len(sim_perm), obj = 1.0/len(sim_perm), lb=-10e10, vtype="C", name='revenue')

    model.update()

    for idx, sim in enumerate(sim_perm):
        p = p_sim['sim_%d'%sim[0]]
        w = w_sim['sim_%d'%sim[1]]
        l = l_sim['sim_%d'%sim[2]]
        f[idx] = model.addVars(settings.T, vtype="B", name='fine')

        for t in range(settings.T):
            # model.addConstr((b[t] + p[t] + w[t] + g[t]*settings.G) >= settings.MIN_GEN_LIST[t], "Min_Generation(%s)"%t)

            model.addConstr((param_dict['min_gen_list'][t] - (p[t] + w[t] + param_dict['turb_cap']*g[t])) <= settings.M*f[idx][t], 'Fine_1')
            model.addConstr((param_dict['min_gen_list'][t] - (p[t] + w[t] + param_dict['turb_cap']*g[t])) >= -1*settings.M*(1-f[idx][t]), 'Fine_2')

        model.addConstr((revenue[idx] == quicksum(l[t]*(p[t] + w[t] + param_dict['turb_cap']*g[t]) - (param_dict['CG']*g[t])\
                        - param_dict['S']*s[t] - f[idx][t]*param_dict['fine'] for t in range(settings.T))), name='set_revenue(%s)'%idx)

    for t in range(1,settings.T):
        model.addConstr(g[t] - g[t-1] <= s[t], "Startup Criteria(%s)"%t)

    # set initial conditions for g[0]
    model.addConstr(g[0] == 1, "Turbine Init Cond")
    model.update()
    ## write out to file
    # model.write('VPP_stochastic.lp')

    # Run
    model.Params.OutputFlag = 0 # 0 = silent mode
    model.optimize()

    return model,g,s,f,revenue


def get_stochastic_nb_output(param_dict):
    p,w,l = get_input_data(param_dict)
    model,g,s,f,revenue = optimize_vpp_stochastic_no_battery(param_dict)

    df = pd.DataFrame(list(range(settings.T)), columns=['hour'])
    df['batt_flow'] = [0]*settings.T
    df['batt_energy'] = [0]*settings.T
    df['turb_on'] = [int(g[t].X) for t in range(settings.T)]
    df['turb_start'] = [int(s[t].X) for t in range(settings.T)]
    df['turb_gen'] = df['turb_on'] * settings.G
    df['solar_gen'] = [round(p[t],2) for t in range(settings.T)]
    df['wind_gen'] = [round(w[t],2) for t in range(settings.T)]
    df['net_gen'] = df['solar_gen'] + df['turb_gen'] + df['wind_gen']
    df['net_out'] = df['net_gen']
    df['elect_price'] = [round(l[t],2) for t in range(settings.T)]
    df['hourly_profit'] = df['elect_price']*df['net_out'] - param_dict['CG']*df['turb_on']\
                    - param_dict['S']*df['turb_start']
    df['fine'] = [0 if (df['net_out'][t] >= param_dict['min_gen_list'][t]) else param_dict['fine']\
                        for t in range(settings.T)]

    rev = []
    fine = []
    for sim in range(param_dict['sims']**3):
        rev.append(round(revenue[sim].X,2))
        fine.append(sum(f[sim][t].X for t in range(settings.T)))

    return df,rev,fine


##### STATIC, NO BATTERY #####

def optimize_vpp_no_battery(param_dict):
    p,w,l = get_input_data(param_dict)
    model = Model("VPP")

    g,s,f = {},{},{}

    for t in range(settings.T):
        g[t] = model.addVar(vtype="B", name="g(%s)"%(t))  # gas turbine running?
        s[t] = model.addVar(vtype="B", name="s(%s)"%(t))  # gas turbine started?
        f[t] = model.addVar(vtype="B", name='fine')  #fine
    model.update()

    # Set objective
    obj = LinExpr()
    obj = quicksum(l[t]*(p[t] + w[t] + param_dict['turb_cap']*g[t]) - (param_dict['CG']*g[t]) - (param_dict['S']*s[t])\
                    - f[t]*param_dict['fine'] for t in range(settings.T))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    for t in range(settings.T):
        # model.addConstr((p[t] + w[t] + g[t]*settings.G) >= settings.MIN_GEN_LIST[t], "Min_Generation(%s)"%t)
        model.addConstr((param_dict['min_gen_list'][t] - (p[t] + w[t] + param_dict['turb_cap']*g[t])) <= settings.M*f[t], 'Fine_1')
        model.addConstr((param_dict['min_gen_list'][t] - (p[t] + w[t] + param_dict['turb_cap']*g[t])) >= -1*settings.M*(1-f[t]), 'Fine_2')

    for t in range(1,settings.T):
        model.addConstr(g[t] - g[t-1] <= s[t], "Startup Criteria(%s)"%t)

    # set initial conditions for g[0]
    model.addConstr(g[0] == 1, "Turbine Init Cond")
    model.update()

    # Run
    model.Params.OutputFlag = 0 # 0 = silent mode
    model.optimize()

    return model,g,s,f


def get_static_nb_output(param_dict):
    p,w,l = get_input_data(param_dict)
    model,g,s,f = optimize_vpp_no_battery(param_dict)

    df = pd.DataFrame(list(range(settings.T)), columns=['hour'])
    df['batt_flow'] = [0]*settings.T
    df['batt_energy'] = [0]*settings.T
    df['turb_on'] = [int(g[t].X) for t in range(settings.T)]
    df['turb_start'] = [int(s[t].X) for t in range(settings.T)]
    df['turb_gen'] = (df['turb_on'] * settings.G).round(2)
    df['solar_gen'] = [round(p[t],2) for t in range(settings.T)]
    df['wind_gen'] = [round(w[t],2) for t in range(settings.T)]
    df['net_gen'] = df['solar_gen'] + df['turb_gen'] + df['wind_gen']
    df['net_out'] = df['net_gen']
    df['elect_price'] = [round(l[t],2) for t in range(settings.T)]
    df['hourly_profit'] = df['elect_price']*df['net_out'] - param_dict['CG']*df['turb_on']\
                    - param_dict['S']*df['turb_start']
    df['fine'] = [round(f[t].X*param_dict['fine'],2) for t in range(settings.T)]
    return df
