import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
import numpy as np
import sys
from numpy import mean as m
import os
import os.path
import sys
import statistics
import glob
from HelperFunctions import get_intact_columns_constant_pressure
from HelperFunctions import get_intact_columns_constant_temperature
from HelperFunctions import plot_shear_stress_vs_normal_stress
from HelperFunctions import get_fitted_plots_constant_temperature
from HelperFunctions import get_fitted_plots_constant_pressure
from HelperFunctions import get_average_shear_normal_stress_and_average_mu_constant_temperature
from HelperFunctions import get_average_shear_normal_stress_and_average_mu_constant_pressure
from mpl_toolkits import mplot3d as Axes3D
from HelperFunctions import plot_shear_stress_vs_normal_stress, plot_variation_in_mu
from HelperFunctions import get_MATLABFIT_dissociation_rates
from HelperFunctions import plot_variation_in_shear_stress_constanttemp
import Dissociation_Rates
from scipy.stats.distributions import t

Temperatures = ["400K"]
Pressures = ['1GPa', '2GPa', '3GPa', '4GPa', '5GPa']

Temperatures = ["400K", "500K", "600K", "700K"]
Pressures = ['2GPa', '3GPa', '4GPa', '5GPa']

#Big_Dataframe_300K = get_intact_columns_constant_temperature("300K", Pressures)
Big_Dataframe_400K = get_intact_columns_constant_temperature("400K", Pressures)
Big_Dataframe_500K = get_intact_columns_constant_temperature("500K", Pressures)
Big_Dataframe_600K = get_intact_columns_constant_temperature("600K", Pressures)
Big_Dataframe_700K = get_intact_columns_constant_temperature("700K", Pressures)

Big_Dataframe_1GPa = get_intact_columns_constant_pressure('1GPa', Temperatures)
Big_Dataframe_2GPa = get_intact_columns_constant_pressure('2GPa', Temperatures)
Big_Dataframe_3GPa = get_intact_columns_constant_pressure('3GPa', Temperatures)
Big_Dataframe_4GPa = get_intact_columns_constant_pressure('4GPa', Temperatures)
Big_Dataframe_5GPa = get_intact_columns_constant_pressure('5GPa', Temperatures)

Big_Dataframe_1GPa.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/OneGPaComparison.csv')
Big_Dataframe_2GPa.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/TwoGPaComparison.csv')
Big_Dataframe_3GPa.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/ThreeGPaComparison.csv')
Big_Dataframe_4GPa.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/FourGPaComparison.csv')
Big_Dataframe_5GPa.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/FiveGPaComparison.csv')

#Big_Dataframe_300K.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/300KComparison.csv')
Big_Dataframe_400K.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/400KComparison.csv')
Big_Dataframe_500K.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/500KComparison.csv')
Big_Dataframe_600K.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/600KComparison.csv')
Big_Dataframe_600K.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/600KComparison.csv')
Big_Dataframe_700K.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/700KComparison.csv')

Temperatures = ["500K", "600K", "700K"]

############## Get Timestep lists for use with MATLAB dissociation rate function ######################

#Timestep_300K = Big_Dataframe_300K['Timestep']
Timestep_400K = Big_Dataframe_400K['Timestep']
Timestep_500K = Big_Dataframe_500K['Timestep']
Timestep_600K = Big_Dataframe_600K['Timestep']
Timestep_700K = Big_Dataframe_700K['Timestep']
Timestep_1GPa = Big_Dataframe_1GPa['Timestep']
Timestep_2GPa = Big_Dataframe_2GPa['Timestep']
Timestep_3GPa = Big_Dataframe_3GPa['Timestep']
Timestep_4GPa = Big_Dataframe_4GPa['Timestep']
Timestep_5GPa = Big_Dataframe_5GPa['Timestep']

Index = 0 # Index = 0 for actual rate, 1 for upper bound, 2 for lower bound

# Dissociation_Rate_300K_1GPa, LogRate_300K_1GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_1GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_2GPa, LogRate_300K_2GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_2GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_3GPa, LogRate_300K_3GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_3GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_4GPa, LogRate_300K_4GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_4GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_5GPa, LogRate_300K_5GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_5GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rates_300K = [Dissociation_Rate_300K_1GPa, Dissociation_Rate_300K_2GPa, Dissociation_Rate_300K_3GPa, Dissociation_Rate_300K_4GPa, Dissociation_Rate_300K_5GPa]
# Log_Rates_300K = [LogRate_300K_1GPa, LogRate_300K_2GPa, LogRate_300K_3GPa, LogRate_300K_4GPa, LogRate_300K_5GPa]


Cutoff = 0.5  #Decimal value between 0 and 1 representing proportial of exeriment you want rates to be calculated out of

Dissociation_Rate_400K_1GPa, LogRate_400K_1GPa = get_MATLABFIT_dissociation_rates(Timestep_400K, Dissociation_Rates.Dissociation_Rate_400K_1GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_400K_2GPa, LogRate_400K_2GPa = get_MATLABFIT_dissociation_rates(Timestep_400K, Dissociation_Rates.Dissociation_Rate_400K_2GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_400K_3GPa, LogRate_400K_3GPa = get_MATLABFIT_dissociation_rates(Timestep_400K, Dissociation_Rates.Dissociation_Rate_400K_3GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_400K_4GPa, LogRate_400K_4GPa = get_MATLABFIT_dissociation_rates(Timestep_400K, Dissociation_Rates.Dissociation_Rate_400K_4GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_400K_5GPa, LogRate_400K_5GPa = get_MATLABFIT_dissociation_rates(Timestep_400K, Dissociation_Rates.Dissociation_Rate_400K_5GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rates_400K = [Dissociation_Rate_400K_1GPa, Dissociation_Rate_400K_2GPa, Dissociation_Rate_400K_3GPa, Dissociation_Rate_400K_4GPa, Dissociation_Rate_400K_5GPa]
Log_Rates_400K = [LogRate_400K_1GPa, LogRate_400K_2GPa, LogRate_400K_3GPa, LogRate_400K_4GPa, LogRate_400K_5GPa]

Dissociation_Rate_500K_1GPa, LogRate_500K_1GPa = get_MATLABFIT_dissociation_rates(Timestep_500K, Dissociation_Rates.Dissociation_Rate_500K_1GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_500K_2GPa, LogRate_500K_2GPa = get_MATLABFIT_dissociation_rates(Timestep_500K, Dissociation_Rates.Dissociation_Rate_500K_2GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_500K_3GPa, LogRate_500K_3GPa = get_MATLABFIT_dissociation_rates(Timestep_500K, Dissociation_Rates.Dissociation_Rate_500K_3GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_500K_4GPa, LogRate_500K_4GPa = get_MATLABFIT_dissociation_rates(Timestep_500K, Dissociation_Rates.Dissociation_Rate_500K_4GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_500K_5GPa, LogRate_500K_5GPa = get_MATLABFIT_dissociation_rates(Timestep_500K, Dissociation_Rates.Dissociation_Rate_500K_5GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rates_500K = [Dissociation_Rate_500K_1GPa, Dissociation_Rate_500K_2GPa, Dissociation_Rate_500K_3GPa, Dissociation_Rate_500K_4GPa, Dissociation_Rate_500K_5GPa]
Log_Rates_500K = [LogRate_500K_1GPa, LogRate_500K_2GPa, LogRate_500K_3GPa, LogRate_500K_4GPa, LogRate_500K_5GPa]

Dissociation_Rate_600K_1GPa, LogRate_600K_1GPa = get_MATLABFIT_dissociation_rates(Timestep_600K, Dissociation_Rates.Dissociation_Rate_600K_1GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_600K_2GPa, LogRate_600K_2GPa = get_MATLABFIT_dissociation_rates(Timestep_600K, Dissociation_Rates.Dissociation_Rate_600K_2GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_600K_3GPa, LogRate_600K_3GPa = get_MATLABFIT_dissociation_rates(Timestep_600K, Dissociation_Rates.Dissociation_Rate_600K_3GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_600K_4GPa, LogRate_600K_4GPa = get_MATLABFIT_dissociation_rates(Timestep_600K, Dissociation_Rates.Dissociation_Rate_600K_4GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_600K_5GPa, LogRate_600K_5GPa = get_MATLABFIT_dissociation_rates(Timestep_600K, Dissociation_Rates.Dissociation_Rate_600K_5GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rates_600K = [Dissociation_Rate_600K_1GPa, Dissociation_Rate_600K_2GPa, Dissociation_Rate_600K_3GPa, Dissociation_Rate_600K_4GPa, Dissociation_Rate_600K_5GPa]
Log_Rates_600K = [LogRate_600K_1GPa, LogRate_600K_2GPa, LogRate_600K_3GPa, LogRate_600K_4GPa, LogRate_600K_5GPa]

Dissociation_Rate_700K_1GPa, LogRate_700K_1GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_1GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_700K_2GPa, LogRate_700K_2GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_2GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_700K_3GPa, LogRate_700K_3GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_3GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_700K_4GPa, LogRate_700K_4GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_4GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rate_700K_5GPa, LogRate_700K_5GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_5GPa_Coefficients[Index], Cutoff=Cutoff)
Dissociation_Rates_700K = [Dissociation_Rate_700K_1GPa, Dissociation_Rate_700K_2GPa, Dissociation_Rate_700K_3GPa, Dissociation_Rate_700K_4GPa, Dissociation_Rate_700K_5GPa]
Log_Rates_700K = [LogRate_700K_1GPa, LogRate_700K_2GPa, LogRate_700K_3GPa, LogRate_700K_4GPa, LogRate_700K_5GPa]

Dissociation_Rates_1GPa = [Dissociation_Rates_400K[0], Dissociation_Rates_500K[0], Dissociation_Rates_600K[0], Dissociation_Rates_700K[0]]
Dissociation_Rates_2GPa = [Dissociation_Rates_400K[1], Dissociation_Rates_500K[1], Dissociation_Rates_600K[1], Dissociation_Rates_700K[1]]
Dissociation_Rates_3GPa = [Dissociation_Rates_400K[2], Dissociation_Rates_500K[2], Dissociation_Rates_600K[2], Dissociation_Rates_700K[2]]
Dissociation_Rates_4GPa = [Dissociation_Rates_400K[3], Dissociation_Rates_500K[3], Dissociation_Rates_600K[3], Dissociation_Rates_700K[3]]
Dissociation_Rates_5GPa = [Dissociation_Rates_400K[4], Dissociation_Rates_500K[4], Dissociation_Rates_600K[4], Dissociation_Rates_700K[4]]

Log_Rates_1GPa = [LogRate_400K_1GPa, LogRate_500K_1GPa, LogRate_600K_1GPa, LogRate_700K_1GPa]
Log_Rates_2GPa = [LogRate_400K_2GPa, LogRate_500K_2GPa, LogRate_600K_2GPa, LogRate_700K_2GPa]
Log_Rates_3GPa = [LogRate_400K_3GPa, LogRate_500K_3GPa, LogRate_600K_3GPa, LogRate_700K_3GPa]
Log_Rates_4GPa = [LogRate_400K_4GPa, LogRate_500K_4GPa, LogRate_600K_4GPa, LogRate_700K_4GPa]
Log_Rates_5GPa = [LogRate_400K_5GPa, LogRate_500K_5GPa, LogRate_600K_5GPa, LogRate_700K_5GPa]


EquilibriumFactor = [-21, -1] # How many rows (out of 99) to ignore before calculating shear stress/friction coefficient, as it won't stabilise until after a certain number of timesteps

def get_average_shear_normal_stress_and_average_mu_constant_temperature(Temperature, Pressures, EquilibriumFactor):
    Friction_Coefficient_Dataframe_Unnamed = pd.read_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/{Temperature}/1GPa/'
                                'fc_ave.dump'.format(Temperature=Temperature), sep=' ')
    Friction_Coefficient_Dataframe = Friction_Coefficient_Dataframe_Unnamed.rename(columns={'v_s_bot' : 'Shear Stress 1GPa', 'v_p_bot' : 'Normal Stress 1GPa'})

    for P in Pressures:
        Dataframe = pd.read_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/{}/{}/'
                                'fc_ave.dump'.format(Temperature, P), sep=' ')
        Big_DataframeP = Dataframe.rename(columns= {'Timestep': 'Timestep {}'.format(P),
                                                        'v_s_bot': 'Shear Stress {}'.format(P),
                                                        'v_p_bot': 'Normal Stress {}'.format(P)})

        Friction_Coefficient_Dataframe = pd.concat([Friction_Coefficient_Dataframe, Big_DataframeP], axis =1)
        Friction_Coefficient_Dataframe = Friction_Coefficient_Dataframe.dropna()


    #print(Friction_Coefficient_Dataframe)
    Mu_Final_Dataframe = Friction_Coefficient_Dataframe.iloc[:, [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14]]
    Mu_Final_Dataframe = Mu_Final_Dataframe.iloc[EquilibriumFactor[0]:EquilibriumFactor[1], :]
    #print(Mu_Final_Dataframe)

    ShearStressMeans = Mu_Final_Dataframe[['Shear Stress 1GPa', 'Shear Stress 2GPa', 'Shear Stress 3GPa', 'Shear Stress 4GPa', 'Shear Stress 5GPa']].mean()
    Average_Shear_Stress_Dictionary = ShearStressMeans.to_dict()
    #print(ShearStressMeans)
    NormalStressMeans = Mu_Final_Dataframe[['Normal Stress 1GPa', 'Normal Stress 2GPa', 'Normal Stress 3GPa', 'Normal Stress 4GPa', 'Normal Stress 5GPa']].mean()
    NormalStressMeans = NormalStressMeans.to_dict()
    #print(NormalStressMeans)


    Average_Mu_Dictionary = {}

    NormalStressMeansList = list(NormalStressMeans.values())
    Normal_Stress = NormalStressMeans.get('Normal Stress 1GPa')
    Shear_Stress = ShearStressMeans.get('Shear Stress 1GPa')
    Average_Mu = Shear_Stress / Normal_Stress
    Average_Mu_Dictionary.update({'Average Mu 1GPa': Average_Mu})

    for P in Pressures:

        Normal_Stress = NormalStressMeans.get('Normal Stress {}'.format(P))
        Shear_Stress = ShearStressMeans.get('Shear Stress {}'.format(P))
        Average_Mu = Shear_Stress / Normal_Stress
        Average_Mu_Dictionary.update({'Average Mu {}'.format(P): Average_Mu})


    Average_Shear_Stress_List = list(Average_Shear_Stress_Dictionary.values())
    #print(Average_Shear_Stress_List)
    Average_Mu_List = list(Average_Mu_Dictionary.values())
    Average_Shear_Stress_List = [x / 10000 for x in Average_Shear_Stress_List] # Conversion to GPa
    NormalStressMeansList = [x/10000 for x in NormalStressMeansList]
    #print(Average_Shear_Stress_List)

    return Average_Shear_Stress_List, Average_Mu_List, NormalStressMeansList

######## Getting Average Shear Stress, Friction Coefficient and Normal Stress #################

Average_Shear_Stress_List_400K, Average_Mu_List_400K, NormalStressMeans_400K = get_average_shear_normal_stress_and_average_mu_constant_temperature("400K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)
Average_Shear_Stress_List_500K, Average_Mu_List_500K, NormalStressMeans_500K = get_average_shear_normal_stress_and_average_mu_constant_temperature("500K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)
Average_Shear_Stress_List_600K, Average_Mu_List_600K, NormalStressMeans_600K = get_average_shear_normal_stress_and_average_mu_constant_temperature("600K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)
Average_Shear_Stress_List_700K, Average_Mu_List_700K, NormalStressMeans_700K = get_average_shear_normal_stress_and_average_mu_constant_temperature("700K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)

Average_Mu_List_1GPa = [Average_Mu_List_400K[0], Average_Mu_List_500K[0], Average_Mu_List_600K[0]]
Average_Mu_List_2GPa = [Average_Mu_List_400K[1], Average_Mu_List_500K[1], Average_Mu_List_600K[1]]
Average_Mu_List_3GPa = [Average_Mu_List_400K[2], Average_Mu_List_500K[2], Average_Mu_List_600K[2]]
Average_Mu_List_4GPa = [Average_Mu_List_400K[3], Average_Mu_List_500K[3], Average_Mu_List_600K[3]]
Average_Mu_List_5GPa = [Average_Mu_List_400K[4], Average_Mu_List_500K[4], Average_Mu_List_600K[4]]

Average_Shear_Stress_List_1GPa = [Average_Shear_Stress_List_400K[0], Average_Shear_Stress_List_500K[0], Average_Shear_Stress_List_600K[0]]
Average_Shear_Stress_List_2GPa = [Average_Shear_Stress_List_400K[1], Average_Shear_Stress_List_500K[1], Average_Shear_Stress_List_600K[1]]
Average_Shear_Stress_List_3GPa = [Average_Shear_Stress_List_400K[2], Average_Shear_Stress_List_500K[2], Average_Shear_Stress_List_600K[2]]
Average_Shear_Stress_List_4GPa = [Average_Shear_Stress_List_400K[3], Average_Shear_Stress_List_500K[3], Average_Shear_Stress_List_600K[3]]
Average_Shear_Stress_List_5GPa = [Average_Shear_Stress_List_400K[4], Average_Shear_Stress_List_500K[4], Average_Shear_Stress_List_600K[4]]

#plot_shear_stress_vs_normal_stress(Average_Shear_Stress_List_400K, Average_Shear_Stress_List_500K, Average_Shear_Stress_List_600K, Average_Shear_Stress_List_700K,
#                                  "400K", "500K", "600K", "700K")

########################## Plotting ln(Rates) vs Shear Stress #####################################
x = np.array([0, 1, 2, 3, 4, 5])
params400K = np.polyfit(Average_Shear_Stress_List_400K, Log_Rates_400K, 1)
params500K = np.polyfit(Average_Shear_Stress_List_500K, Log_Rates_500K, 1)
params600K = np.polyfit(Average_Shear_Stress_List_600K, Log_Rates_600K, 1)
params700K = np.polyfit(Average_Shear_Stress_List_700K, Log_Rates_700K, 1)

RatesvsShear, RvS3 = plt.subplots()
RvS3.set_title('Log of Dissociation Rates vs Shear Stress')
RvS3.set_xlabel('Shear Stress(GPa)')
RvS3.set_ylabel('Log of Dissociation Rate ($ns^-1$)')
RvS3.scatter(Average_Shear_Stress_List_400K, Log_Rates_400K)
RvS3.scatter(Average_Shear_Stress_List_500K, Log_Rates_500K)
RvS3.scatter(Average_Shear_Stress_List_600K, Log_Rates_600K)
RvS3.scatter(Average_Shear_Stress_List_700K, Log_Rates_700K)
RvS3.plot(x, params400K[0] * x + params400K[1], label='400K Fitted')
RvS3.plot(x, params500K[0] * x + params500K[1], label='500K Fitted')
RvS3.plot(x, params600K[0] * x + params600K[1], label='600K Fitted')
RvS3.plot(x, params700K[0] * x + params700K[1], label='700K Fitted')
RvS3.set_xlim(0, 2)
RvS3.set_ylim(0, 3.5)
RvS3.legend(loc='lower right')
plt.show()

####### Calculate Activation Volume, Using  Carlos' conversion to get in Angstrom^3 #################

activation_vol_400K = (params400K[0]) * (1.38065) * 400 * 1e-2
activation_vol_500K = (params500K[0]) * (1.38065) * 500 * 1e-2
activation_vol_600K = (params600K[0]) * (1.38065) * 600 * 1e-2
activation_vol_700K = (params700K[0]) * (1.38065) * 700 * 1e-2

alpha = 0.05

def linear(x, m, n):
    return m * x + n

coef_sh_cf, coef_sh_pcov = optimize.curve_fit(linear, Average_Shear_Stress_List_700K, Log_Rates_700K)

sigma = coef_sh_pcov[0, 0] ** 0.5
dof = max(0, len(Log_Rates_700K) - len(params700K))
#print(dof)

tval = t.ppf(1.0 - alpha / 2., dof)
uncert = sigma * tval
uncert400 = uncert * (1.38065) * 400 * 1e-2
uncert500 = uncert * (1.38065) * 500 * 1e-2
uncert600 = uncert * (1.38065) * 600 * 1e-2
uncert700 = uncert * (1.38065) * 700 * 1e-2
# coefs_sh_cf.append(coef_sh_cf[0] * (1.38065) * useT * 1e-2)
# coefs_sh_cf_uncert.append(uncert)
print(f'Activation Volume uncertainty at 400K is {uncert400}')
print(f'Activation Volume uncertainty at 500K is {uncert500}')
print(f'Activation Volume uncertainty at 600K is {uncert600}')
print(f'Activation Volume uncertainty at 700K is {uncert700}')

Activation_Volumes = [activation_vol_400K, activation_vol_500K, activation_vol_600K, activation_vol_700K]
mean_actv = np.average(Activation_Volumes)

print('Activation Volume 400K = ' + str(activation_vol_400K))
print('Activation Volume 500K = ' + str(activation_vol_500K))
print('Activation Volume 600K = ' + str(activation_vol_600K))
print('Activation Volume 700K = ' + str(activation_vol_700K))

############ Plotting lnk vs 1000/T  #########################

Temperatures = [400, 500, 600, 700]
Inverse_Temperatures = np.array([1/x for x in Temperatures])

trend1GPa = np.polyfit(Inverse_Temperatures, Log_Rates_1GPa, 1)
trend2GPa = np.polyfit(Inverse_Temperatures, Log_Rates_2GPa, 1)
trend3GPa = np.polyfit(Inverse_Temperatures, Log_Rates_3GPa, 1)
trend4GPa = np.polyfit(Inverse_Temperatures, Log_Rates_4GPa, 1)
trend5GPa = np.polyfit(Inverse_Temperatures, Log_Rates_5GPa, 1)

fig1, ax1 = plt.subplots()
ax1.set_title('Log of Dissociation Rates against Inverse of Temperatures')
ax1.set_xlabel('1000/T (K-1)')
ax1.set_ylabel('ln(Rate) (ns-1)')
#ax1.set_xlim([1.4, 3.4])
#ax1.set_ylim([0.5, 6])
ax1.scatter(Inverse_Temperatures, Log_Rates_1GPa)
ax1.scatter(Inverse_Temperatures, Log_Rates_2GPa)
ax1.scatter(Inverse_Temperatures, Log_Rates_3GPa)
ax1.scatter(Inverse_Temperatures, Log_Rates_4GPa)
ax1.scatter(Inverse_Temperatures, Log_Rates_5GPa)
ax1.legend()

Fit1GPa = np.poly1d(trend1GPa)
Fit2GPa = np.poly1d(trend2GPa)
Fit3GPa = np.poly1d(trend3GPa)
Fit4GPa = np.poly1d(trend4GPa)
Fit5GPa = np.poly1d(trend5GPa)

ax1.plot(Inverse_Temperatures, Fit1GPa(Inverse_Temperatures), label='1GPa')
ax1.plot(Inverse_Temperatures, Fit2GPa(Inverse_Temperatures), label='2GPa')
ax1.plot(Inverse_Temperatures, Fit3GPa(Inverse_Temperatures), label='3GPa')
ax1.plot(Inverse_Temperatures, Fit4GPa(Inverse_Temperatures), label='4GPa')
ax1.plot(Inverse_Temperatures, Fit5GPa(Inverse_Temperatures), label='5GPa')
ax1.legend()
#plt.show()

ActivationEnergy_1GPa = (((1 * 1e9 * (np.average(Average_Mu_List_1GPa) * mean_actv) * 1e-30) - 1.381 * trend1GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_2GPa = (((2 * 1e9 * (np.average(Average_Mu_List_2GPa) * mean_actv) * 1e-30) - 1.381 * trend2GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_3GPa = (((3 * 1e9 * (np.average(Average_Mu_List_3GPa) * mean_actv) * 1e-30) - 1.381 * trend3GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_4GPa = (((4 * 1e9 * (np.average(Average_Mu_List_4GPa) * mean_actv) * 1e-30) - 1.381 * trend4GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_5GPa = (((5 * 1e9 * (np.average(Average_Mu_List_5GPa) * mean_actv) * 1e-30) - 1.381 * trend5GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000

lnA1GPa = np.log(trend1GPa[1] * (10 ** 9))
lnA2GPa = np.log(trend2GPa[1] * (10 ** 9))
lnA3GPa = np.log(trend3GPa[1] * (10 ** 9))
lnA4GPa = np.log(trend4GPa[1] * (10 ** 9))
lnA5GPa = np.log(trend5GPa[1] * (10 ** 9))

print(f"ln(A) at 1GPa is {lnA1GPa}")
print(f"ln(A) at 2GPa is {lnA2GPa}")
print(f"ln(A) at 3GPa is {lnA3GPa}")
print(f"ln(A) at 4GPa is {lnA4GPa}")
print(f"ln(A) at 5GPa is {lnA5GPa}")


print('Activation Energy 1GPa =' + str(ActivationEnergy_1GPa))
print('Activation Energy 2GPa =' + str(ActivationEnergy_2GPa))
print('Activation Energy 3GPa =' + str(ActivationEnergy_3GPa))
print('Activation Energy 4GPa =' + str(ActivationEnergy_4GPa))
print('Activation Energy 5GPa =' + str(ActivationEnergy_5GPa))

def linear(x, m, n):
    return m * x + n

coef_p_cf_1GPa, coef_p_pcov_1GPa = optimize.curve_fit(linear, Inverse_Temperatures, Log_Rates_1GPa)
sigma_A1GPa = coef_p_pcov_1GPa[1, 1] ** 0.5
sigma_m1GPa = coef_p_pcov_1GPa[0, 0] ** 0.5
dof = max(0, len(Log_Rates_1GPa) - len(coef_p_cf_1GPa))

alpha = 0.05

tval = t.ppf(1.0 - alpha / 2., 4.1532)
sigma = np.std(Activation_Volumes)
error_actv = sigma * tval / np.sqrt(len(Activation_Volumes))
#print(error_actv)

uncert_A1GPa = sigma_A1GPa * tval
uncert_m1GPa = sigma_m1GPa * tval

coef_p_cf_2GPa, coef_p_pcov_2GPa = optimize.curve_fit(linear, Inverse_Temperatures, Log_Rates_2GPa)
sigma_A2GPa = coef_p_pcov_2GPa[1, 1] ** 0.5
sigma_m2GPa = coef_p_pcov_2GPa[0, 0] ** 0.5
dof = max(0, len(Log_Rates_2GPa) - len(coef_p_cf_2GPa))
uncert_A2GPa = sigma_A2GPa * tval
uncert_m2GPa = sigma_m2GPa * tval

coef_p_cf_3GPa, coef_p_pcov_3GPa = optimize.curve_fit(linear, Inverse_Temperatures, Log_Rates_3GPa)
sigma_A3GPa = coef_p_pcov_3GPa[1, 1] ** 0.5
sigma_m3GPa = coef_p_pcov_3GPa[0, 0] ** 0.5
dof = max(0, len(Log_Rates_3GPa) - len(coef_p_cf_3GPa))
uncert_A3GPa = sigma_A3GPa * tval
uncert_m3GPa = sigma_m3GPa * tval

coef_p_cf_4GPa, coef_p_pcov_4GPa = optimize.curve_fit(linear, Inverse_Temperatures, Log_Rates_4GPa)
sigma_A4GPa = coef_p_pcov_4GPa[1, 1] ** 0.5
sigma_m4GPa = coef_p_pcov_4GPa[0, 0] ** 0.5
dof = max(0, len(Log_Rates_4GPa) - len(coef_p_cf_4GPa))
uncert_A4GPa = sigma_A4GPa * tval
uncert_m4GPa = sigma_m4GPa * tval

coef_p_cf_5GPa, coef_p_pcov_5GPa = optimize.curve_fit(linear, Inverse_Temperatures, Log_Rates_5GPa)
sigma_A5GPa = coef_p_pcov_5GPa[1, 1] ** 0.5
sigma_m5GPa = coef_p_pcov_5GPa[0, 0] ** 0.5
dof = max(0, len(Log_Rates_5GPa) - len(coef_p_cf_5GPa))
uncert_A5GPa = sigma_A5GPa * tval
uncert_m5GPa = sigma_m5GPa * tval

ActivationEnergy_1GPa = (((1 * 1e9 * (np.average(Average_Mu_List_1GPa) * mean_actv) * 1e-30) - 1.381 * trend1GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_2GPa = (((2 * 1e9 * (np.average(Average_Mu_List_2GPa) * mean_actv) * 1e-30) - 1.381 * trend2GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_3GPa = (((3 * 1e9 * (np.average(Average_Mu_List_3GPa) * mean_actv) * 1e-30) - 1.381 * trend3GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_4GPa = (((4 * 1e9 * (np.average(Average_Mu_List_4GPa) * mean_actv) * 1e-30) - 1.381 * trend4GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_5GPa = (((5 * 1e9 * (np.average(Average_Mu_List_5GPa) * mean_actv) * 1e-30) - 1.381 * trend5GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000

err_E_1GPa = (np.sqrt(((1 * 1e9 * np.average(Average_Mu_List_1GPa) * error_actv * 1e-30) ** 2 + (1.381 * uncert_m1GPa * 1e-23) ** 2)) * (6.02214076 * (10**23))) / 1000
err_E_2GPa = (np.sqrt(((2 * 1e9 * np.average(Average_Mu_List_2GPa) * error_actv * 1e-30) ** 2 + (1.381 * uncert_m2GPa * 1e-23) ** 2)) * (6.02214076 * (10**23))) / 1000
err_E_3GPa = (np.sqrt(((3 * 1e9 * np.average(Average_Mu_List_3GPa) * error_actv * 1e-30) ** 2 + (1.381 * uncert_m3GPa * 1e-23) ** 2)) * (6.02214076 * (10**23))) / 1000
err_E_4GPa = (np.sqrt(((4 * 1e9 * np.average(Average_Mu_List_4GPa) * error_actv * 1e-30) ** 2 + (1.381 * uncert_m4GPa * 1e-23) ** 2)) * (6.02214076 * (10**23))) / 1000
err_E_5GPa = (np.sqrt(((5 * 1e9 * np.average(Average_Mu_List_5GPa) * error_actv * 1e-30) ** 2 + (1.381 * uncert_m5GPa * 1e-23) ** 2)) * (6.02214076 * (10**23))) / 1000

print(f"Error for Activation Energy at 1GPa = {err_E_1GPa}")
print(f"Error for Activation Energy at 2GPa = {err_E_2GPa}")
print(f"Error for Activation Energy at 3GPa = {err_E_3GPa}")
print(f"Error for Activation Energy at 4GPa = {err_E_4GPa}")
print(f"Error for Activation Energy at 5GPa = {err_E_5GPa}")

uncertlnA1GPa = np.log(uncert_A1GPa * (10 ** 9))
uncertlnA2GPa = np.log(uncert_A2GPa * (10 ** 9))
uncertlnA3GPa = np.log(uncert_A3GPa * (10 ** 9))
uncertlnA4GPa = np.log(uncert_A4GPa * (10 ** 9))
uncertlnA5GPa = np.log(uncert_A5GPa * (10 ** 9))

print(f"Error for ln A at 1GPa = {uncert_A1GPa}")
print(f"Error for ln A at 2GPa = {uncert_A2GPa}")
print(f"Error for ln A at 3GPa = {uncert_A3GPa}")
print(f"Error for ln A at 4GPa = {uncert_A4GPa}")
print(f"Error for ln A at 5GPa = {uncert_A5GPa}")

pressures = ['2GPa', '3GPa', '4GPa', '5GPa']
#plot_variation_in_shear_stress_constanttemp('600K', pressures)

########################## Plotting ln(Rates) vs Normal Stress #####################################
x = np.array([0, 1, 2, 3, 4, 5])
params400K = np.polyfit(NormalStressMeans_400K, Log_Rates_400K, 1)
params500K = np.polyfit(NormalStressMeans_500K, Log_Rates_500K, 1)
params600K = np.polyfit(NormalStressMeans_600K, Log_Rates_600K, 1)
params700K = np.polyfit(NormalStressMeans_700K, Log_Rates_700K, 1)

RatesvsNormal, RvN3 = plt.subplots()
RvN3.set_title('Log of Dissociation Rates vs Normal Stress - Alpha Fe')
RvN3.set_xlabel('Normal Stress(GPa)')
RvN3.set_ylabel('Log of Dissociation Rate (ns-1)')
RvN3.scatter(NormalStressMeans_400K, Log_Rates_400K)
RvN3.scatter(NormalStressMeans_500K, Log_Rates_500K)
RvN3.scatter(NormalStressMeans_600K, Log_Rates_600K)
RvN3.scatter(NormalStressMeans_700K, Log_Rates_700K)
RvN3.plot(x, params400K[0] * x + params400K[1], label='400K Fitted')
RvN3.plot(x, params500K[0] * x + params500K[1], label='500K Fitted')
RvN3.plot(x, params600K[0] * x + params600K[1], label='600K Fitted')
RvN3.plot(x, params700K[0] * x + params700K[1], label='700K Fitted')
#RvN3.set_xlim(1, 5)
# RvN3.set_ylim(0, 25)
RvN3.legend(loc='lower right')
##plt.show()



def function(data, a, b, c):
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c) #TODO change fitting function

x_data = []
y_data = []
z_data = []

data = [[400, Average_Shear_Stress_List_400K[0], LogRate_400K_1GPa], [400, Average_Shear_Stress_List_400K[1], LogRate_400K_2GPa], [400, Average_Shear_Stress_List_400K[2], LogRate_400K_3GPa], [400, Average_Shear_Stress_List_400K[3], LogRate_400K_4GPa], [400, Average_Shear_Stress_List_400K[4], LogRate_400K_5GPa],
        [500, Average_Shear_Stress_List_500K[0], LogRate_500K_1GPa], [500, Average_Shear_Stress_List_500K[1], LogRate_500K_2GPa], [500, Average_Shear_Stress_List_500K[2], LogRate_500K_3GPa], [500, Average_Shear_Stress_List_500K[3], LogRate_500K_4GPa], [500, Average_Shear_Stress_List_500K[4], LogRate_500K_5GPa],
        [600, Average_Shear_Stress_List_600K[0], LogRate_600K_1GPa], [600, Average_Shear_Stress_List_600K[1], LogRate_600K_2GPa], [600, Average_Shear_Stress_List_600K[2], LogRate_600K_3GPa], [600, Average_Shear_Stress_List_600K[3], LogRate_600K_4GPa], [600, Average_Shear_Stress_List_600K[4], LogRate_600K_5GPa],
        [700, Average_Shear_Stress_List_700K[0], LogRate_700K_1GPa], [700, Average_Shear_Stress_List_700K[1], LogRate_700K_2GPa], [700, Average_Shear_Stress_List_700K[2], LogRate_700K_3GPa], [700, Average_Shear_Stress_List_700K[3], LogRate_700K_4GPa], [700, Average_Shear_Stress_List_700K[4], LogRate_700K_5GPa]]

for item in data:
    x_data.append(item[0])
    y_data.append(item[1])
    z_data.append(item[2])


parameters, covariance = optimize.curve_fit(function, [x_data, y_data], z_data)

# create surface function model
# setup data points for calculating surface model
model_x_data = np.linspace(min(x_data), max(x_data), 40)
model_y_data = np.linspace(min(y_data), max(y_data), 40)

# create coordinate arrays for vectorized evaluations
X, Y = np.meshgrid(model_x_data, model_y_data)
# calculate Z coordinate array
Z = function(np.array([X, Y]), *parameters)

z = []
for row in Z:
    row.sort()
    z.append(row)

zlogs = np.array(z)
# print(Z)
# print('####################')
# print(z)

import matplotlib.cm
cm = plt.get_cmap("jet")


# setup figure object
fig = plt.figure()
# setup 3d object
ax4 = plt.axes(projection='3d')
# plot surface
ax4.plot_surface(X, Y, Z, cmap=cm, alpha=0.5, edgecolor='black', linewidth=0.3)
ax4.set_title('3D Plot - Variation in Log of Dissociation Rates - Alpha Fe')
# plot input data
ax4.scatter(x_data, y_data, z_data, color='black', alpha=1)
# set plot descriptions
ax4.set_xlabel('Temperature (K)')
ax4.invert_xaxis()
ax4.set_ylabel('Shear Stress (GPa)')
ax4.set_zlabel('Log of Dissociation Rate (per ns)')
plt.show()
plt.close(fig)

def function(data, a, b, c):
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c) #TODO change fitting function

x_data = []
y_data = []
z_data = []

data = [[400, Average_Shear_Stress_List_400K[0], Dissociation_Rate_400K_1GPa], [400, Average_Shear_Stress_List_400K[1], Dissociation_Rate_400K_2GPa], [400, Average_Shear_Stress_List_400K[2], Dissociation_Rate_400K_3GPa], [400, Average_Shear_Stress_List_400K[3], Dissociation_Rate_400K_4GPa], [400, Average_Shear_Stress_List_400K[4], Dissociation_Rate_400K_5GPa],
        [500, Average_Shear_Stress_List_500K[0], Dissociation_Rate_500K_1GPa], [500, Average_Shear_Stress_List_500K[1], Dissociation_Rate_500K_2GPa], [500, Average_Shear_Stress_List_500K[2], Dissociation_Rate_500K_3GPa], [500, Average_Shear_Stress_List_500K[3], Dissociation_Rate_500K_4GPa], [500, Average_Shear_Stress_List_500K[4], Dissociation_Rate_500K_5GPa],
        [600, Average_Shear_Stress_List_600K[0], Dissociation_Rate_600K_1GPa], [600, Average_Shear_Stress_List_600K[1], Dissociation_Rate_600K_2GPa], [600, Average_Shear_Stress_List_600K[2], Dissociation_Rate_600K_3GPa], [600, Average_Shear_Stress_List_600K[3], Dissociation_Rate_600K_4GPa], [600, Average_Shear_Stress_List_600K[4], Dissociation_Rate_600K_5GPa],
        [700, Average_Shear_Stress_List_700K[0], Dissociation_Rate_700K_1GPa], [700, Average_Shear_Stress_List_700K[1], Dissociation_Rate_700K_2GPa], [700, Average_Shear_Stress_List_700K[2], Dissociation_Rate_700K_3GPa], [700, Average_Shear_Stress_List_700K[3], Dissociation_Rate_700K_4GPa], [700, Average_Shear_Stress_List_700K[4], Dissociation_Rate_700K_5GPa]]

for item in data:
    x_data.append(item[0])
    y_data.append(item[1])
    z_data.append(item[2])

parameters, covariance = optimize.curve_fit(function, [x_data, y_data], z_data)

# create surface function model
# setup data points for calculating surface model
model_x_data = np.linspace(min(x_data), max(x_data), 40)
model_y_data = np.linspace(min(y_data), max(y_data), 40)
# create coordinate arrays for vectorized evaluations
X, Y = np.meshgrid(model_x_data, model_y_data)
# calculate Z coordinate array
Z = function(np.array([X, Y]), *parameters)

z = []
for row in Z:
    row.sort()
    z.append(row)

z = np.array(z)
# print(Z)
# print('####################')
# print(z)

import matplotlib.cm
cm = plt.get_cmap("jet")


# setup figure object
fig = plt.figure()
# setup 3d object
ax4 = plt.axes(projection='3d')
# plot surface
ax4.plot_surface(X, Y, Z, cmap=cm, alpha=0.5, edgecolor= 'black', linewidth=0.3)
ax4.set_title('3D Plot - Variation in Dissociation Rates - Alpha Fe')
# plot input data
ax4.scatter(x_data, y_data, z_data, color='black', alpha=1)
# set plot descriptions
ax4.set_xlabel('Temperature (K)')
ax4.invert_xaxis()
ax4.set_ylabel('Shear Stress (GPa)')
ax4.set_zlabel('Dissociation Rate (per ns)')
plt.show()

########## CALCULATING THE ACTIVATION ENERGY, VOLUME AND PREFACTOR FROM 3D FIT ##########

logRates3D = zlogs
shearstresses = model_y_data

Index = 0
sigma = []
params = []
while Index < len(shearstresses) - 1:
    for logRates3Drow in logRates3D:
        coef_sh_cf, coef_sh_pcov = optimize.curve_fit(linear, shearstresses, logRates3Drow)
        paramsrow = np.polyfit(shearstresses, logRates3Drow, 1)
        sigmarow = coef_sh_pcov[0, 0] ** 0.5
        sigma.append(sigmarow)
        params.append(paramsrow[0])
        Index +=1

sigma = np.array(sigma)
params = np.array(params)
params = np.average(params)
sigma = np.average(sigma)

activation_vol_3D = (params) * (1.38065) * 700 * 1e-2
print(f'3D Activation Volume is {activation_vol_3D}')

alpha = 0.05
sigma = sigma ** 0.5
dof = 2
tval = t.ppf(1.0 - alpha / 2., dof)
uncert = sigma * tval
uncert_ActivationVolume = uncert * (1.38065) * 500 * 1e-2
print(f'Activation Volume uncertainty for 3D fit is {uncert_ActivationVolume}')

"""
Below, we calculate the activation energies. Use the activation volume from above, using the meshgrid
data to get the correct shape for the shear forces and our calculated Z-matrix as the
"""

logRates3D = zlogs
temperatures = model_x_data

inverse_temperatures = [1 / x for x in temperatures]

Index = 0
sigma = []
SigmaA = []
params = []
interceptaverage = []
while Index < len(inverse_temperatures) - 1:
    for logRates3Drow in logRates3D:
        coef_sh_cf, coef_sh_pcov = optimize.curve_fit(linear, inverse_temperatures, logRates3Drow)
        paramsrow = np.polyfit(inverse_temperatures, logRates3Drow, 1)
        sigmarow = coef_sh_pcov[0, 0] ** 0.5
        sigma_A = coef_sh_pcov[1, 1] ** 0.5
        sigma.append(sigmarow)
        SigmaA.append(sigma_A)
        params.append(paramsrow[0])
        intercept = paramsrow[1]
        interceptaverage.append((intercept))
        Index +=1

sigma = np.array(sigma)
params = np.array(params)
SigmaA = np.array(SigmaA)
interceptaverage = np.array(interceptaverage)
params = np.average(params)
sigma = np.average(sigma)
interceptaverage = np.average(interceptaverage)
alpha = 0.05
sigma = sigma ** 0.5
tval = t.ppf(1.0 - alpha / 2., dof)
uncert = sigma * tval

Mu1GPa = np.average(Average_Mu_List_1GPa)
Mu2GPa = np.average(Average_Mu_List_2GPa)
Mu3GPa = np.average(Average_Mu_List_3GPa)
Mu4GPa = np.average(Average_Mu_List_4GPa)
Mu5GPa = np.average(Average_Mu_List_5GPa)

MuAveragesDifferentPressures = np.array([Mu1GPa, Mu2GPa, Mu3GPa, Mu4GPa, Mu5GPa])
AverageMu = np.average(MuAveragesDifferentPressures)

ActivationEnergy_3D = (((3 * 1e9 * (AverageMu * activation_vol_3D) * 1e-30) - 1.381 * params * 1e-23) * 6.02214076 * (10**23)) / 1000

print(f'Activation Energy for 3D fit is {ActivationEnergy_3D}')

error_3D = (np.sqrt((3 * 1e9 * np.average(AverageMu) * uncert_ActivationVolume * 1e-30) ** 2 + (1.381 * uncert * 1e-23) ** 2) * 6.02214076 * (10**23)) / 1000
print(f"Activation_Energy Error Is {error_3D}")

uncert_prefactor_3D = sigma_A * tval
print("ln(A) for 3D fit is " + str(np.log(interceptaverage * (10 ** 9))))
print(f"ln(A) uncertainty is {uncert_prefactor_3D}" + str(10 ** 9))

######### Plotting 3D fit vs 2D fit results along with their error margins ##########
"""
Need to get a list with:
- Values for Activation Volumes at different temperatures
- Values for Activation Energies at different pressures
- Values for ln(A) at different pressures
- List of errors for each of the above quantities
- Make a graph for each surface chemistry
- Eventually will need to do the same for the different sliding speeds

"""
Temperatures = [400, 500, 600, 700]
Pressures = [1, 2, 3, 4, 5]


Activation_Energies = [ActivationEnergy_1GPa, ActivationEnergy_2GPa, ActivationEnergy_3GPa, ActivationEnergy_4GPa, ActivationEnergy_5GPa]
Activation_Energy_Errors = [err_E_1GPa, err_E_2GPa, err_E_3GPa, err_E_4GPa, err_E_5GPa]
ActivationEnergy_3D = ActivationEnergy_3D
ActivationEnergy_3D_error = error_3D
ActivationEnergy_3D_error_UpperBound_Value = float(float(ActivationEnergy_3D) + float(ActivationEnergy_3D_error))
ActivationEnergy_3D_error_LowerBound_Value = float(float(ActivationEnergy_3D) - float(ActivationEnergy_3D_error))

Activation_Energy_Error_Plot, Ea2Dvs3d  = plt.subplots()
Ea2Dvs3d.set_title('Comparison of Activation Energies from 2D and 3D Fits - FeC')
Ea2Dvs3d.set_xlabel('Normal Stress(GPa)')
Ea2Dvs3d.set_ylabel('Activation Energy')
Ea2Dvs3d.scatter(Pressures, Activation_Energies)
Ea2Dvs3d.errorbar(Pressures, Activation_Energies, yerr=Activation_Energy_Errors, linestyle="None", fmt='o', capsize=3)
Ea2Dvs3d.axhline(y=ActivationEnergy_3D)
Pressures = [0.5, 1, 2, 3, 4, 5, 5.5]
Ea2Dvs3d.fill_between(Pressures, ActivationEnergy_3D_error_LowerBound_Value, ActivationEnergy_3D_error_UpperBound_Value, alpha=0.4)
Ea2Dvs3d.set_xlim(0.5, 5.5)
Ea2Dvs3d.set_ylim(0, 35)
plt.show()

################ Activation Volume Errors #############################

Activation_Volumes = [activation_vol_400K, activation_vol_500K, activation_vol_600K, activation_vol_700K]
Activation_Volume_Errors = [uncert400, uncert500, uncert600, uncert700]
Activation_Volume_3D = activation_vol_3D
Activation_Volume_3D_Error = uncert_ActivationVolume
ActivationVolume_3D_error_UpperBound_Value = float(float(Activation_Volume_3D) + float(Activation_Volume_3D_Error))
ActivationVolume_3D_error_LowerBound_Value = float(float(Activation_Volume_3D) - float(Activation_Volume_3D_Error))

Activation_Volume_Error_Plot, Av2Dvs3d  = plt.subplots()
Av2Dvs3d.set_title('Comparison of Activation Volumes from 2D and 3D Fits - FeC')
Av2Dvs3d.set_xlabel('Normal Stress(GPa)')
Av2Dvs3d.set_ylabel('Activation Volume')
Av2Dvs3d.scatter(Temperatures, Activation_Volumes)
Av2Dvs3d.errorbar(Temperatures, Activation_Volumes, yerr=Activation_Volume_Errors, linestyle="None", fmt='o', capsize=3)
Av2Dvs3d.axhline(y=Activation_Volume_3D)
Temperatures = [350, 400, 500, 600, 700, 750]
Av2Dvs3d.fill_between(Temperatures, ActivationVolume_3D_error_LowerBound_Value, ActivationVolume_3D_error_UpperBound_Value, alpha=0.4)
Av2Dvs3d.set_xlim(350, 750)
Av2Dvs3d.set_ylim(0, 30)
plt.show()

#################### Prefactor Errors #########################
Pressures = [1, 2, 3, 4, 5]
Prefactors = [lnA1GPa, lnA2GPa, lnA3GPa, lnA4GPa, lnA5GPa]
PrefactorErrors = [uncert_A1GPa, uncert_A2GPa, uncert_A3GPa, uncert_A4GPa, uncert_A5GPa]
lnA_3D = np.log(interceptaverage * (10 ** 9))
lnA_3D_error = uncert_prefactor_3D
print(lnA_3D_error)
lnA_3D_error_UpperBound_Value = float(float(lnA_3D) + float(lnA_3D_error))
lnA_3D_error_LowerBound_Value = float(float(lnA_3D) - float(lnA_3D_error))

Prefactor_Error_Plot, lnA2Dvs3d  = plt.subplots()
lnA2Dvs3d.set_title('Comparison of Prefactors from 2D and 3D Fits - FeC')
lnA2Dvs3d.set_xlabel('Normal Stress(GPa)')
lnA2Dvs3d.set_ylabel('Prefactor')
lnA2Dvs3d.scatter(Pressures, Prefactors)
lnA2Dvs3d.errorbar(Pressures, Prefactors, yerr=PrefactorErrors, linestyle="None", fmt='o', capsize=3)
lnA2Dvs3d.axhline(y=lnA_3D)
Pressures = [0.5, 1, 2, 3, 4, 5, 5.5]
lnA2Dvs3d.fill_between(Pressures, lnA_3D_error_LowerBound_Value, lnA_3D_error_UpperBound_Value, alpha=0.4)
lnA2Dvs3d.set_xlim(0.5, 5.5)
lnA2Dvs3d.set_ylim(18, 28)
plt.show()
