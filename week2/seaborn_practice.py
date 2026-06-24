
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
names = sns.get_dataset_names()
#   print('DATASET NAMES ', names)  #     PIRNT ALL THE BUILT IN DATASET OF SEABRON LIBRARY
penguins = sns.load_dataset('penguins')
#   print(penguins.head())
#   print(penguins["species"].value_counts())
# penguins = penguins.dropna(subset=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])  #CLEANING THE DATA
#   print(penguins.head())
#   sns.set_style('whitegrid')

#   sns.scatterplot(data = penguins, x = 'flipper_length_mm', y = 'body_mass_g', hue = 'island', palette = 'Set2')  |SCATTERPLOT |
#   sns.stripplot( data =penguins, x = 'species', y = 'body_mass_g', hue = 'island', dodge = True , )       |STRIPLOT |

#   sns.swarmplot( data =penguins, x = 'species', y = 'body_mass_g', hue = 'island', dodge = True , )       | SWARM PLOT |

#   sns.set_context("notebook")
#   sns.histplot(data = penguins, x= 'body_mass_g')         | HISTOGRAM PLOT |

#   sns.regplot(data = penguins, x = 'body_mass_g', y = 'flipper_length_mm')        | REGRESSION PLOT |

#   sns.lineplot(data = penguins, x = 'body_mass_g', y = 'flipper_length_mm' )

#   sns.jointplot(data = penguins, x = 'body_mass_g', y = 'flipper_length_mm',  kind = 'reg')

#   sns.countplot(data = penguins, x = 'species', hue = 'sex')

# sns.boxplot(data = penguins, x = 'species', y = 'body_mass_g', hue = 'island')



graph = sns.PairGrid(data = penguins, hue = 'sex')
graph.map_upper(sns.scatterplot)    #UPPER TRIANGLE
graph.map_lower(sns.kdeplot)        #LOWER TRIANGLE
graph.map_diag(sns.histplot)        #DIGONAL


plt.show()