## Define and package Constants

T = 168
HOURS = [h for h in range(T)]
MIN_GEN = 12
FINE = 1500
V_MAX = 240
B_MAX = 40
B_MIN = -40
G = 52
W_MAX = 61.5
P_MAX = 24
B_EFF = 0.938
CG = 2600 # 0.42 efficiency; assuming constant temperature
S = (1/6)*CG # startup time, spindown time
SEASON = 0  # 0 for january, 1 for July

M = 10e10
MIN_GEN_LIST=([B_MIN]*6 + [MIN_GEN]*15 + [B_MIN]*3)*7

SOLAR_DATA_FILE = 'data/solar_data_2017.csv'
WIND_DATA_FILE = 'data/wind_data_2017.csv'
ENERGY_DATA_FILE_JAN = 'data/Hourly_Energy_Price_Jan_2017.csv'
ENERGY_DATA_FILE_JUL = 'data/Hourly_Energy_Price_Jul_2017.csv'

SIMS = 5

param_dict = {
    'mod': 'stoc_vpp',
    'season': SEASON,
    'sims': SIMS,
    'V_MAX': V_MAX,
    'B_MAX': B_MAX,
    'S': S,
    'CG': CG,
    'min_gen_list': MIN_GEN_LIST,
    'fine': FINE,
    'solar_cap': P_MAX,
    'wind_cap': W_MAX,
    'turb_cap': G,
    'sig_p': 2,
    'sig_w': 4,
    'sig_l': 3,
}
