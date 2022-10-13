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


def plot_normal_stress_vs_ln_dissociation_rates(*NormalStressList, **LogDissoicationRatesList):
    print(NormalStressList)
    print(LogDissoicationRatesList)

plot_normal_stress_vs_ln_dissociation_rates()