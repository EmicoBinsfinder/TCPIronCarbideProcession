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
import Dissociation_Rates
from scipy.stats.distributions import t

Temperatures = ["500K", "600K"]
Pressures = ['2GPa', '3GPa', '4GPa', '5GPa']

Big_Dataframe_400K = get_intact_columns_constant_temperature("400K", Pressures)
Big_Dataframe_500K = get_intact_columns_constant_temperature("500K", Pressures)
Big_Dataframe_600K = get_intact_columns_constant_temperature("600K", Pressures)

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
#Big_Dataframe_700K.to_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/700KComparison.csv')

############## Get Timestep lists for use with MATLAB dissociation rate function ######################

#Timestep_300K = Big_Dataframe_300K['Timestep']
Timestep_400K = Big_Dataframe_400K['Timestep']
Timestep_500K = Big_Dataframe_500K['Timestep']
Timestep_600K = Big_Dataframe_600K['Timestep']
#Timestep_700K = Big_Dataframe_700K['Timestep']
Timestep_1GPa = Big_Dataframe_1GPa['Timestep']
Timestep_2GPa = Big_Dataframe_2GPa['Timestep']
Timestep_3GPa = Big_Dataframe_3GPa['Timestep']
Timestep_4GPa = Big_Dataframe_4GPa['Timestep']
Timestep_5GPa = Big_Dataframe_5GPa['Timestep']

Cutoff = None
Index = 0 # Index = 0 for actual rate, 1 for upper bound, 2 for lower bound

activation_volume_error_300K = []
activation_volume_error_400K = []
activation_volume_error_500K = []
activation_volume_error_600K = []
activation_volume_error_700K = []

activation_energy_errors_1GPa = []
activation_energy_errors_2GPa = []
activation_energy_errors_3GPa = []
activation_energy_errors_4GPa = []
activation_energy_errors_5GPa = []

prefactor_error_1GPa = []
prefactor_error_2GPa = []
prefactor_error_3GPa = []
prefactor_error_4GPa = []
prefactor_error_5GPa = []

activation_volumes = []
activation_energies = []
lnA = []

# Dissociation_Rate_300K_1GPa, LogRate_300K_1GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_1GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_2GPa, LogRate_300K_2GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_2GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_3GPa, LogRate_300K_3GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_3GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_4GPa, LogRate_300K_4GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_4GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_300K_5GPa, LogRate_300K_5GPa = get_MATLABFIT_dissociation_rates(Timestep_300K, Dissociation_Rates.Dissociation_Rate_300K_5GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rates_300K = [Dissociation_Rate_300K_1GPa, Dissociation_Rate_300K_2GPa, Dissociation_Rate_300K_3GPa, Dissociation_Rate_300K_4GPa, Dissociation_Rate_300K_5GPa]
# Log_Rates_300K = [LogRate_300K_1GPa, LogRate_300K_2GPa, LogRate_300K_3GPa, LogRate_300K_4GPa, LogRate_300K_5GPa]

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

# Dissociation_Rate_700K_1GPa, LogRate_700K_1GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_1GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_700K_2GPa, LogRate_700K_2GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_2GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_700K_3GPa, LogRate_700K_3GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_3GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_700K_4GPa, LogRate_700K_4GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_4GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rate_700K_5GPa, LogRate_700K_5GPa = get_MATLABFIT_dissociation_rates(Timestep_700K, Dissociation_Rates.Dissociation_Rate_700K_5GPa_Coefficients[Index], Cutoff=Cutoff)
# Dissociation_Rates_700K = [Dissociation_Rate_700K_1GPa, Dissociation_Rate_700K_2GPa, Dissociation_Rate_700K_3GPa, Dissociation_Rate_700K_4GPa, Dissociation_Rate_700K_5GPa]
# Log_Rates_700K = [LogRate_700K_1GPa, LogRate_700K_2GPa, LogRate_700K_3GPa, LogRate_700K_4GPa, LogRate_700K_5GPa]

Dissociation_Rates_1GPa = [Dissociation_Rates_400K[0], Dissociation_Rates_500K[0], Dissociation_Rates_600K[0]]
Dissociation_Rates_2GPa = [Dissociation_Rates_400K[1], Dissociation_Rates_500K[1], Dissociation_Rates_600K[1]]
Dissociation_Rates_3GPa = [Dissociation_Rates_400K[2], Dissociation_Rates_500K[2], Dissociation_Rates_600K[2]]
Dissociation_Rates_4GPa = [Dissociation_Rates_400K[3], Dissociation_Rates_500K[3], Dissociation_Rates_600K[3]]
Dissociation_Rates_5GPa = [Dissociation_Rates_400K[4], Dissociation_Rates_500K[4], Dissociation_Rates_600K[4]]

Log_Rates_1GPa = [LogRate_400K_1GPa, LogRate_500K_1GPa, LogRate_600K_1GPa]
Log_Rates_2GPa = [LogRate_400K_2GPa, LogRate_500K_2GPa, LogRate_600K_2GPa]
Log_Rates_3GPa = [LogRate_400K_3GPa, LogRate_500K_3GPa, LogRate_600K_3GPa]
Log_Rates_4GPa = [LogRate_400K_4GPa, LogRate_500K_4GPa, LogRate_600K_4GPa]
Log_Rates_5GPa = [LogRate_400K_5GPa, LogRate_500K_5GPa, LogRate_600K_5GPa]

# print(f"Dissociation Rate at 300K, 1GPa is {Dissociation_Rate_300K_1GPa}, log of dissociation rate is {LogRate_300K_1GPa}")
# print(f"Dissociation Rate at 300K, 2GPa is {Dissociation_Rate_300K_2GPa}, log of dissociation rate is {LogRate_300K_2GPa}")
# print(f"Dissociation Rate at 300K, 3GPa is {Dissociation_Rate_300K_3GPa}, log of dissociation rate is {LogRate_300K_3GPa}")
# print(f"Dissociation Rate at 300K, 4GPa is {Dissociation_Rate_300K_4GPa}, log of dissociation rate is {LogRate_300K_4GPa}")
# print(f"Dissociation Rate at 300K, 5GPa is {Dissociation_Rate_300K_5GPa}, log of dissociation rate is {LogRate_300K_5GPa}")
print(f"Dissociation Rate at 400K, 1GPa is {Dissociation_Rate_400K_1GPa}, log of dissociation rate is {LogRate_400K_1GPa}")
print(f"Dissociation Rate at 400K, 2GPa is {Dissociation_Rate_400K_2GPa}, log of dissociation rate is {LogRate_400K_2GPa}")
print(f"Dissociation Rate at 400K, 3GPa is {Dissociation_Rate_400K_3GPa}, log of dissociation rate is {LogRate_400K_3GPa}")
print(f"Dissociation Rate at 400K, 4GPa is {Dissociation_Rate_400K_4GPa}, log of dissociation rate is {LogRate_400K_4GPa}")
print(f"Dissociation Rate at 400K, 5GPa is {Dissociation_Rate_400K_5GPa}, log of dissociation rate is {LogRate_400K_5GPa}")
print(f"Dissociation Rate at 500K, 1GPa is {Dissociation_Rate_500K_1GPa}, log of dissociation rate is {LogRate_500K_1GPa}")
print(f"Dissociation Rate at 500K, 2GPa is {Dissociation_Rate_500K_2GPa}, log of dissociation rate is {LogRate_500K_2GPa}")
print(f"Dissociation Rate at 500K, 3GPa is {Dissociation_Rate_500K_3GPa}, log of dissociation rate is {LogRate_500K_3GPa}")
print(f"Dissociation Rate at 500K, 4GPa is {Dissociation_Rate_500K_4GPa}, log of dissociation rate is {LogRate_500K_4GPa}")
print(f"Dissociation Rate at 500K, 5GPa is {Dissociation_Rate_500K_5GPa}, log of dissociation rate is {LogRate_500K_5GPa}")
print(f"Dissociation Rate at 600K, 1GPa is {Dissociation_Rate_600K_1GPa}, log of dissociation rate is {LogRate_600K_1GPa}")
print(f"Dissociation Rate at 600K, 2GPa is {Dissociation_Rate_600K_2GPa}, log of dissociation rate is {LogRate_600K_2GPa}")
print(f"Dissociation Rate at 600K, 3GPa is {Dissociation_Rate_600K_3GPa}, log of dissociation rate is {LogRate_600K_3GPa}")
print(f"Dissociation Rate at 600K, 4GPa is {Dissociation_Rate_600K_4GPa}, log of dissociation rate is {LogRate_600K_4GPa}")
print(f"Dissociation Rate at 600K, 5GPa is {Dissociation_Rate_600K_5GPa}, log of dissociation rate is {LogRate_600K_5GPa}")
# print(f"Dissociation Rate at 700K, 1GPa is {Dissociation_Rate_700K_1GPa}, log of dissociation rate is {LogRate_700K_1GPa}")
# print(f"Dissociation Rate at 700K, 2GPa is {Dissociation_Rate_700K_2GPa}, log of dissociation rate is {LogRate_700K_2GPa}")
# print(f"Dissociation Rate at 700K, 3GPa is {Dissociation_Rate_700K_3GPa}, log of dissociation rate is {LogRate_700K_3GPa}")
# print(f"Dissociation Rate at 700K, 4GPa is {Dissociation_Rate_700K_4GPa}, log of dissociation rate is {LogRate_700K_4GPa}")
# print(f"Dissociation Rate at 700K, 5GPa is {Dissociation_Rate_700K_5GPa}, log of dissociation rate is {LogRate_700K_5GPa}")

EquilibriumFactor = 40 # How many rows (out of 99) to ignore before calculating shear stress/friction coefficient, as it won't stabilise until after a certain number of timesteps

def get_average_shear_normal_stress_and_average_mu_constant_temperature(Temperature, Pressures, EquilibriumFactor):
    Friction_Coefficient_Dataframe_Unnamed = pd.read_csv('F:/PhD/TCPDecompositionExperiments/Completed/FeC/{}/1GPa/'
                                'fc_ave.dump'.format(Temperature), sep=' ')
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
    Mu_Final_Dataframe = Mu_Final_Dataframe.iloc[EquilibriumFactor:, :]
    #print(Mu_Final_Dataframe)

    ShearStressMeans = Mu_Final_Dataframe[['Shear Stress 1GPa', 'Shear Stress 2GPa', 'Shear Stress 3GPa', 'Shear Stress 4GPa', 'Shear Stress 5GPa']].mean()
    Average_Shear_Stress_Dictionary = ShearStressMeans.to_dict()
    #print(ShearStressMeans)
    NormalStressMeans = Mu_Final_Dataframe[['Normal Stress 1GPa', 'Normal Stress 2GPa', 'Normal Stress 3GPa', 'Normal Stress 4GPa', 'Normal Stress 5GPa']].mean()
    NormalStressMeans = NormalStressMeans.to_dict()
    #print(NormalStressMeans)


    Average_Mu_Dictionary = {}

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
    #print(Average_Shear_Stress_List)

    return Average_Shear_Stress_List, Average_Mu_List, NormalStressMeans


######## Getting Average Shear Stress, Friction Coefficient and Normal Stress #################

#Average_Shear_Stress_List_300K, Average_Mu_List_300K, NormalStressMeans_300K = get_average_shear_normal_stress_and_average_mu_constant_temperature("300K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)
Average_Shear_Stress_List_400K, Average_Mu_List_400K, NormalStressMeans_400K = get_average_shear_normal_stress_and_average_mu_constant_temperature("400K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)
Average_Shear_Stress_List_500K, Average_Mu_List_500K, NormalStressMeans_500K = get_average_shear_normal_stress_and_average_mu_constant_temperature("500K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)
Average_Shear_Stress_List_600K, Average_Mu_List_600K, NormalStressMeans_600K = get_average_shear_normal_stress_and_average_mu_constant_temperature("600K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)
#Average_Shear_Stress_List_700K, Average_Mu_List_700K, NormalStressMeans_700K = get_average_shear_normal_stress_and_average_mu_constant_temperature("700K", Pressures=Pressures, EquilibriumFactor=EquilibriumFactor)

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

# print(Average_Shear_Stress_List_300K)
# print(Average_Shear_Stress_List_400K)
# print(Average_Shear_Stress_List_500K)
# print(Average_Shear_Stress_List_600K)
# print(Average_Shear_Stress_List_700K)

plot_shear_stress_vs_normal_stress(Average_Shear_Stress_List_400K, Average_Shear_Stress_List_500K, Average_Shear_Stress_List_600K,
                                  "400K", "500K", "600K")

########################## Plotting ln(Rates) vs Shear Stress #####################################
x = np.array([0, 1, 2, 3, 4, 5])
#params300K = np.polyfit(Average_Shear_Stress_List_300K, Log_Rates_300K, 1)
params400K = np.polyfit(Average_Shear_Stress_List_400K, Log_Rates_400K, 1)
params500K = np.polyfit(Average_Shear_Stress_List_500K, Log_Rates_500K, 1)
params600K = np.polyfit(Average_Shear_Stress_List_600K, Log_Rates_600K, 1)
#params700K = np.polyfit(Average_Shear_Stress_List_700K, Log_Rates_700K, 1)

RatesvsShear, RvS3 = plt.subplots()
RvS3.set_title('Log of Dissociation Rates vs Shear Stress')
RvS3.set_xlabel('Shear Stress(GPa)')
RvS3.set_ylabel('Log of Dissociation Rate ($ns^-1$)')
#RvS3.scatter(Average_Shear_Stress_List_300K, Log_Rates_300K)
RvS3.scatter(Average_Shear_Stress_List_400K, Log_Rates_400K)
RvS3.scatter(Average_Shear_Stress_List_500K, Log_Rates_500K)
RvS3.scatter(Average_Shear_Stress_List_600K, Log_Rates_600K)
#RvS3.scatter(Average_Shear_Stress_List_700K, Log_Rates_700K)
#RvS3.plot(x, params300K[0] * x + params300K[1], label='300K Fitted')
RvS3.plot(x, params400K[0] * x + params400K[1], label='400K Fitted')
RvS3.plot(x, params500K[0] * x + params500K[1], label='500K Fitted')
RvS3.plot(x, params600K[0] * x + params600K[1], label='600K Fitted')
#RvS3.plot(x, params700K[0] * x + params700K[1], label='700K Fitted')
RvS3.set_xlim(0.5, 2.45)
RvS3.set_ylim(-1, 4.5)
RvS3.legend(loc='lower right')
plt.show()

####### Calculate Activation Volume, Using  Carlos' conversion to get in Angstrom^3 #################

# activation_vol_300K = (params300K[0]) * (1.38065) * 300 * 1e-2
# activation_volume_error_300K.append(activation_vol_300K)
activation_vol_400K = (params400K[0]) * (1.38065) * 400 * 1e-2

activation_vol_500K = (params500K[0]) * (1.38065) * 500 * 1e-2

activation_vol_600K = (params600K[0]) * (1.38065) * 600 * 1e-2

# activation_vol_700K = (params700K[0]) * (1.38065) * 700 * 1e-2
# activation_volume_error_700K.append(activation_vol_700K)

Activation_Volumes = [activation_vol_400K, activation_vol_500K, activation_vol_600K]
mean_actv = np.average(Activation_Volumes)

#
# print('Activation Volume 300K = ' + str(activation_vol_300K))
# print('Activation Volume 400K = ' + str(activation_vol_400K))
# print('Activation Volume 500K = ' + str(activation_vol_500K))
# print('Activation Volume 600K = ' + str(activation_vol_600K))
# print('Activation Volume 700K = ' + str(activation_vol_700K))

############ Plotting lnk vs 1000/T  #########################

Temperatures = [400, 500, 600]
Inverse_Temperatures = np.array([1/x for x in Temperatures])

trend1GPa = np.polyfit(Inverse_Temperatures, Log_Rates_1GPa, 1)
trend2GPa = np.polyfit(Inverse_Temperatures, Log_Rates_2GPa, 1)
trend3GPa = np.polyfit(Inverse_Temperatures, Log_Rates_3GPa, 1)
trend4GPa = np.polyfit(Inverse_Temperatures, Log_Rates_4GPa, 1)
trend5GPa = np.polyfit(Inverse_Temperatures, Log_Rates_5GPa, 1)

# print(trend1GPa)
# print(trend2GPa)
# print(trend3GPa)
# print(trend4GPa)
# print(trend5GPa)

fig1, ax1 = plt.subplots()
ax1.set_title('Dissociation Rates against Inverse of Temperatures')
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
plt.show()

ActivationEnergy_1GPa = (((1 * 1e9 * (np.average(Average_Mu_List_1GPa) * mean_actv) * 1e-30) - 1.381 * trend1GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
activation_energy_errors_1GPa.append(ActivationEnergy_1GPa)
ActivationEnergy_2GPa = (((2 * 1e9 * (np.average(Average_Mu_List_2GPa) * mean_actv) * 1e-30) - 1.381 * trend2GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
activation_energy_errors_2GPa.append(ActivationEnergy_2GPa)
ActivationEnergy_3GPa = (((3 * 1e9 * (np.average(Average_Mu_List_3GPa) * mean_actv) * 1e-30) - 1.381 * trend3GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
activation_energy_errors_3GPa.append(ActivationEnergy_3GPa)
ActivationEnergy_4GPa = (((4 * 1e9 * (np.average(Average_Mu_List_4GPa) * mean_actv) * 1e-30) - 1.381 * trend4GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
activation_energy_errors_4GPa.append(ActivationEnergy_4GPa)
ActivationEnergy_5GPa = (((5 * 1e9 * (np.average(Average_Mu_List_5GPa) * mean_actv) * 1e-30) - 1.381 * trend5GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
activation_energy_errors_5GPa.append(ActivationEnergy_5GPa)

print("ln(A) at 1GPa is " + str(np.log(trend1GPa[1] * (10 ** 9))))
print("ln(A) at 2GPa is " + str(np.log(trend2GPa[1] * (10 ** 9))))
print("ln(A) at 3GPa is " + str(np.log(trend3GPa[1] * (10 ** 9))))
print("ln(A) at 4GPa is " + str(np.log(trend4GPa[1] * (10 ** 9))))
print("ln(A) at 5GPa is " + str(np.log(trend5GPa[1] * (10 ** 9))))

print('Activation Energy 1GPa =' + str(ActivationEnergy_1GPa))
print('Activation Energy 2GPa =' + str(ActivationEnergy_2GPa))
print('Activation Energy 3GPa =' + str(ActivationEnergy_3GPa))
print('Activation Energy 4GPa =' + str(ActivationEnergy_4GPa))
print('Activation Energy 5GPa =' + str(ActivationEnergy_5GPa))

# xdata = [300, 400, 500, 600, 700]
# ydata = [1, 2, 3, 4, 5]
#
# Temperatures = ["300K", "400K", "500K", "600K", "700K"]
# df = pd.DataFrame(columns=[Temperatures])
# df.loc[len(df)] = Average_Shear_Stress_List_1GPa
# df.loc[len(df)] = Average_Shear_Stress_List_2GPa
# df.loc[len(df)] = Average_Shear_Stress_List_3GPa
# df.loc[len(df)] = Average_Shear_Stress_List_4GPa
# df.loc[len(df)] = Average_Shear_Stress_List_5GPa

def linear(x, m, n):
    return m * x + n

coef_p_cf_1GPa, coef_p_pcov_1GPa = optimize.curve_fit(linear, Inverse_Temperatures, Log_Rates_1GPa)
sigma_A = coef_p_pcov_1GPa[1, 1] ** 0.5
sigma_m = coef_p_pcov_1GPa[0, 0] ** 0.5
dof = max(0, len(Log_Rates_1GPa) - len(coef_p_cf_1GPa))

alpha = 0.05

tval = t.ppf(1.0 - alpha / 2., 4.1532)
sigma = np.std(Activation_Volumes)
error_actv = sigma * tval / np.sqrt(len(Activation_Volumes))
print(error_actv)

uncert_A = sigma_A * tval
uncert_m = sigma_m * tval

ActivationEnergy_1GPa = (((1 * 1e9 * (np.average(Average_Mu_List_1GPa) * mean_actv) * 1e-30) - 1.381 * trend1GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_2GPa = (((2 * 1e9 * (np.average(Average_Mu_List_2GPa) * mean_actv) * 1e-30) - 1.381 * trend2GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_3GPa = (((3 * 1e9 * (np.average(Average_Mu_List_3GPa) * mean_actv) * 1e-30) - 1.381 * trend3GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_4GPa = (((4 * 1e9 * (np.average(Average_Mu_List_4GPa) * mean_actv) * 1e-30) - 1.381 * trend4GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000
ActivationEnergy_5GPa = (((5 * 1e9 * (np.average(Average_Mu_List_5GPa) * mean_actv) * 1e-30) - 1.381 * trend5GPa[0] * 1e-23) * 6.02214076 * (10**23)) / 1000

err_E_1GPa = np.sqrt((5 * 1e9 * np.average(Average_Mu_List_5GPa) * error_actv * 1e-30) ** 2 + (1.381 * uncert_m * 1e-23) ** 2)

print("Activation_Energy Error Is " + str((float(err_E_1GPa)) * 6.02214076 * (10**23) / 1000))


