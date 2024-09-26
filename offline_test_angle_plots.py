import torch
from nn_models import GeneralModel
import numpy as np
import matplotlib.pyplot as plt
import math
from testers import Tester
import os
current_dir = os.getcwd()


# this code outputs plots that shows offline test of our model and ground truth for each of 7 joint angles

test=Tester()
for inum in range(500):
    y1_model,y2_model,y3_model,y4_model,y5_model,y6_model,y7_model=test.get_delta_ang_offline(False,num=inum)
    # y1_base,y2_base,y3_base,y4_base,y5_base,y6_base,y7_base=test.get_delta_ang_offline(False,num=inum)
    indexes=test.get_changes_indexes(inum)
    print(inum)

    x=[i for i in range(test.traj_step_size)]
    y1_real,y2_real,y3_real,y4_real,y5_real,y6_real,y7_real=test.get_real_delta_ang(num=inum,use_angle=False)



    plt.figure(figsize=(15, 12))  # Adjust figure size as per your preference


    # Creating subplots
    for i in range(1, 8):  # Loop through each plot from 1 to 7
        
        plt.subplot(4, 2, i)  # Creating a 4x2 grid of subplots, activate subplot i
        for k in indexes:
            plt.axvline(x = k, color = 'r')
        plt.axvline(x = 20, color = 'purple')
        plt.scatter(x, globals()[f'y{i}_model'], color='green', label='model',s=1)  # Scatter plot
        # plt.scatter(x, globals()[f'y{i}_base'], color='green', label='base',s=1)  # Scatter plot
        plt.scatter(x, globals()[f'y{i}_real'], color='blue', label='ground truth',s=1)  # Scatter plot

        plt.xlabel('Step number')
        plt.ylabel('Delta joint angle')
        plt.title(f'Delta Joint Angle of {i} in {inum}th traj')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()  # Adjust spacing between plots
    directory = 'offline_train0'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the filename for the plot
    plot_filename = f'{directory}/plot_{inum}.png'

    # Save the plot
    plt.savefig(plot_filename)
    # plt.show()
    plt.close() 
    plt.show()




