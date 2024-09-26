import torch
import rospy
from nn_models import GeneralModel, MLPBaseline
from std_msgs.msg import String
import socket
import time
import numpy as np
from torkin import TorKin
from threading import Lock
import torch.nn as nn
import matplotlib.pyplot as plt
from testers import Tester
from udp_comm import Comm
import os
comm = Comm()
current_dir = os.getcwd()


def online_test(use_baseline,dataset,traj_num):
    state_size = 14
    step_size = 1
    target_size=6
    onehot_size=2
    joint_size = 7
    if use_baseline:
        baseline = MLPBaseline(20, 7)
        baseline.load_state_dict(torch.load(current_dir+ '/weights/815_trajs_static_mse_los_tar_cart_base_500_135.519K_params/train_no_0/fbc_700.pth'))
        baseline.eval()

    else:
        general_model = GeneralModel(state_size, target_size+onehot_size, joint_size, False)
        general_model.load_state_dict(torch.load('/home/deniz/catkin_ws/src/feedback_controller/fbc/neural_network/weights/500traj_2obj|mse_los|tar_cart|288.021K_params/train_no_0/fbc_500.pth', map_location=torch.device('cpu')))  # Provide the path to the GeneralModel weights
        general_model.eval()

    point_loss=0

    elem = dataset[traj_num]
    state = torch.tensor(elem[0][1:1 + state_size].tolist())
    
    milestones=t.get_changes_indexes(traj_num)
    print(milestones)
    milestone_js=[torch.tensor(elem[milestones[0]-1][1:8].tolist()),torch.tensor(elem[milestones[1]-1][1:8].tolist())]
    one_hot=elem[0][step_size + state_size + target_size : step_size + state_size + target_size + onehot_size ].tolist()
    goal = elem[0][step_size + state_size : step_size + state_size + target_size ].tolist()



    obstA = torch.tensor(goal[0:3])
    obstB = torch.tensor(goal[3:6])
    # obstC = torch.tensor(goal[6:9])
    comm.move('point1', obstA)
    comm.move('point2', obstB)
    # comm.move('point3', obstC)
    comm.create_and_pub_msg(state[:7])
    rospy.sleep(5)
    path_point = 0
    all_joints=[]


    # Evaluate Baseline Model
    loss = 0
    criterion = nn.L1Loss()
    

    for i in range(150):
        goal_tensor=torch.tensor(goal+one_hot)

        delta = torch.tensor(elem[i][step_size + state_size + target_size + onehot_size: step_size + state_size + target_size + onehot_size + joint_size ].tolist())
        if use_baseline:
            all = torch.cat((goal_tensor, state), dim=0)
            velocities_tensor = baseline(all)

        else:

            velocities_tensor = general_model(goal_tensor, state)[0]

        loss += criterion(delta, velocities_tensor)

        state[:7] += velocities_tensor
        comm.create_and_pub_msg(state[:7])
        rospy.sleep(0.2)
        comm.jsLock.acquire()
        state = torch.cat((torch.tensor((list(comm.joint_state))),velocities_tensor),dim=0)
        comm.jsLock.release()
        all_joints.append(state[:7])
        if one_hot[0]==1:
            comm.move('point2',t.get_end_effector_pos(list(comm.joint_state)))
        # elif one_hot[2]==1:
        #     comm.move('point3', t.get_end_effector_pos(list(comm.joint_state)))
    

        # state += velocities_tensor
        if path_point==0 and t.close_enough(state,milestone_js[0]):
            path_point=1
            one_hot=[1,0]
            print('here1')
        elif path_point==1 and t.close_enough(state,milestone_js[1]):
            print('here2')
            path_point=2
            one_hot=[0,1] 

    point_loss += path_point
    return all_joints, point_loss


t = Tester()

point_loss=0
point_loss_model=0
num=5
dataset = np.load(current_dir+ '/data/torobo/815_trajs_static/train_ds.npy', allow_pickle=True, encoding='latin1')[:]
kin = TorKin()
rospy.init_node('denz')
print(dataset.shape)

# Plot results
def plot_results(all_joints, all_joints_general,plot_number,elem):
    state_size = 14
    step_size = 1
    target_size=6
    onehot_size=2
    joint_size = 7
    goal = elem[0][step_size + state_size : step_size + state_size + target_size ].tolist()


    obstA = torch.tensor(goal[0:3])
    obstB = torch.tensor(goal[3:6])
    # obstC = torch.tensor(goal[6:9])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = [], [], []
    x_general, y_general, z_general = [], [], []

    for new_angles in all_joints:
        my_l = [0, 0]
        for j in new_angles:
            my_l.append(float(j))
        p, R = kin.forwardkin(1, np.array(my_l))
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])

    for new_angles in all_joints_general:
        my_l = [0, 0]
        for j in new_angles:
            my_l.append(float(j))
        p, R = kin.forwardkin(1, np.array(my_l))
        x_general.append(p[0])
        y_general.append(p[1])
        z_general.append(p[2])

    x_real, y_real = t.get_real_coordinates(plot_number)

    ax.scatter(x, y, z, c='g', s=1, label='Baseline Model')
    ax.scatter(x_general, y_general, z_general, c='b', s=1, label='General Model')
    ax.scatter(x_real, y_real, c='r', s=1, label='Ground Truth')

    ax.scatter([obstA[0]], [obstA[1]], [obstA[2]], c='k', marker='o')
    ax.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='k', marker='o')
    # ax.scatter([obstC[0]], [obstC[1]], [obstC[2]], c='k', marker='o')

    ax.text(obstA[0], obstA[1], obstA[2], 'point A', color='black', fontsize=10, ha='center')
    ax.text(obstB[0], obstB[1], obstB[2], 'point B', color='black', fontsize=10, ha='center')
    # ax.text(obstC[0], obstC[1], obstC[2], 'point C', color='black', fontsize=10, ha='center')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Model Trajectories')
    ax.legend()

    directory = 'plots'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the filename for the plot
    plot_filename = f'{directory}/plot_{plot_number}.png'

    # Save the plot
    plt.savefig(plot_filename)

    # Print confirmation message
    print(f'Saved plot to {plot_filename}')


t = Tester()

point_loss=0
point_loss_model=0
num=5
dataset = np.load(current_dir+ '/data/torobo/815_trajs_static/train_ds.npy', allow_pickle=True, encoding='latin1')[:]
kin = TorKin()
for num in range(4,200):

    rospy.init_node('denz')
    print(dataset.shape)

    # Execute evaluation and plotting
    print('waining for model')
    rospy.sleep(5)
    comm.which('\n\n\n\ndnfc start on path'+str(num)+'\n\n\n\n')
    rospy.sleep(5)
    all_joints_general,loss_general=online_test(False,dataset,num)
    # print('waining for baseline')
    # rospy.sleep(5)
    # comm.which('\n\n\n\nbaseline start on path'+str(num)+'\n\n\n\n')
    # rospy.sleep(5)

    # all_joints,loss=online_test(True,dataset,num)


    # plot_results(all_joints, all_joints_general,num,dataset[num])

    # print('hereee')
    # print(loss/(4))
    print(loss_general)

