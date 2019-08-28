import pandas as pd
import settings
import math

# param_dict = settings.param_dict

### SOLAR
# MW = 0.161 (efficiency) *1.66815 (area in m^2) * 75000 (# of panels) * DNI (in W/m^2)
## TODO -- add in settings params to scale solar and wind energy production
def solar_energy(solar_DNI, param_dict):
    n_panels = 75000/24*param_dict['solar_cap']
    # given in MW
    energy = [0.161*1.66815*n_panels/1e6*DNI for DNI in solar_DNI]
    return energy

def get_solar_data(param_dict):
    solar_data = pd.read_csv(settings.SOLAR_DATA_FILE, header = 2)
    solar_DNI = solar_data.loc[:,'DNI']
    solar_DNI_jan = solar_DNI.loc[:settings.T-1]   #first week of July solar data
    solar_DNI_jul = solar_DNI.loc[4344:(4344+167)]  #first week of July solar data
    if param_dict['season'] == 0:
        p = solar_energy(solar_DNI_jan, param_dict)
    elif param_dict['season'] == 1:
        p = solar_energy(solar_DNI_jul, param_dict)
    return p

### WIND

def wind_energy(windspeed, param_dict):
    energy = [] #return in MW
    cap = param_dict['wind_cap']
    n_turbines = cap/1.5
    windspeed_65 = [i*math.log(65/.03)/math.log(10/.03)*2 for i in windspeed]
    for i in windspeed_65:
        if 3 <= i < 7:
            energy.append(n_turbines*(68*i-200)/1000)
        elif 7 <= i < 12:
            energy.append(n_turbines*(243*i-1427)/1000)
        elif 12 <= i < 24:
            energy.append(n_turbines*1500/1000)
        else:
            energy.append(0)
    return energy

def get_wind_data(param_dict):
    weather_data = pd.read_csv(settings.WIND_DATA_FILE, header = 2)
    windspeed_ms = weather_data.loc[:,'Wind Speed']
    windspeed_ms_jan = windspeed_ms.loc[:settings.T-1]   # first week of january wind data
    windspeed_ms_jul = windspeed_ms.loc[4344:(4344+167)]   # first week of july wind data
    if param_dict['season']== 0:
        w = wind_energy(windspeed_ms_jan, param_dict)
    elif param_dict['season'] == 1:
        w = wind_energy(windspeed_ms_jul, param_dict)
    return w


### ELECTRICITY PRICES

def get_price_data(param_dict):
    if param_dict['season'] == 0:
        prices = pd.read_csv(settings.ENERGY_DATA_FILE_JAN)
        l = prices.iloc[7:,14]
        l=l.reset_index(drop=True)
        l.drop(columns=['index'])
    elif param_dict['season'] == 1:
        prices = pd.read_csv(settings.ENERGY_DATA_FILE_JUL)
        l=prices.iloc[6:,14]
        l=l.reset_index(drop=True)
    return l


def get_input_data(param_dict):
    p = get_solar_data(param_dict)
    w = get_wind_data(param_dict)
    l = get_price_data(param_dict)
    return p,w,l
