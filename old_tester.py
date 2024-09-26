import matplotlib.pyplot as plt
import numpy as np
from torkin import TorKin
import torch
from nn_models import GeneralModelSmall as GeneralModelSmall, MLPBaselineSmall, GeneralModelMedium,GeneralModelBig,MLPBaselineMedium,MLPBaselineBig
import torch.nn as nn
import math

class Tester():
    def __init__(self) -> None:
        self.state_size=7
        self.step_size=1
        self.joint_size=7
        general_path='/Users/denizakkoc/Desktop/fbc_new/neural_network'

        self.model = GeneralModelSmall(7,13,7,False)
        self.baseline = MLPBaselineSmall(self.state_size+13,7)
        self.model_path=general_path+'/weights/815_trajs_static|mse_los|tar_cart|v2|360|92.734K_params/train_no_0'
        self.base_path=general_path+'/weights/815_trajs_static|mse_los|tar_cart|base|v2|360|95.319K_params/train_no_0'

        self.epoch1 = '/fbc_5000.pth'
        self.epoch2 = '/fbc_5000.pth'

        self.model.load_state_dict(torch.load(self.model_path+self.epoch1,map_location=torch.device('cpu')))
        self.baseline.load_state_dict(torch.load(self.base_path+self.epoch2,map_location=torch.device('cpu')))
        self.dataset_name='test'
        self.dataset = np.load(general_path+'/data/torobo/815_trajs_static/traj_normalized.npy', allow_pickle=True, encoding='latin1')[:200]

        self.kin=TorKin()

        self.criterion= nn.L1Loss()
        self.criterion2= nn.MSELoss()


    def get_delta_ang_offline(self,usebaseline,num):

        y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[]
        elem=self.dataset[num]

        if usebaseline:
            self.baseline.eval()
        else:
            self.model.eval()


        for i in range(299):
            input_tensor=torch.tensor(elem[i][self.step_size:self.step_size+self.state_size].tolist())
            
            # input_tensor=torch.cat((joint_angles_tensor,velocities_tensor),dim=0)
            goal=elem[i][self.step_size+self.state_size+self.joint_size:].tolist()
            goal_tensor=torch.tensor(goal)
            if usebaseline:
                all=torch.cat((goal_tensor, input_tensor),dim=0)
                velocities_tensor=self.baseline(all)
            else:
                velocities_tensor=self.model(goal_tensor, input_tensor)[0]
                
            y1.append(float(velocities_tensor[0]))
            y2.append(float(velocities_tensor[1]))
            y3.append(float(velocities_tensor[2]))
            y4.append(float(velocities_tensor[3]))
            y5.append(float(velocities_tensor[4]))
            y6.append(float(velocities_tensor[5]))
            y7.append(float(velocities_tensor[6]))


        return y1,y2,y3,y4,y5,y6,y7
    

    def get_emulated(self,usebaseline,num,use_angle=False,return_path_point=False):

        y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[]
        elem=self.dataset[num]
        state=torch.tensor(elem[0][1:1+self.state_size].tolist())

        goal=elem[0][self.step_size+self.state_size+self.joint_size:self.step_size+self.state_size+self.joint_size+9].tolist()
        one_hot=elem[0][self.step_size+self.state_size+self.joint_size+9:].tolist()
        milestones=self.get_changes_indexes(num)
        path_point=0
        print(milestones)
        milestone_js=[torch.tensor(elem[milestones[0]-1][1:8].tolist()),torch.tensor(elem[milestones[1]-1][1:8].tolist()),torch.tensor(elem[milestones[2]-1][1:8].tolist()),torch.tensor(elem[milestones[3]-1][1:8].tolist())]

        if usebaseline:
            self.baseline.eval()
        else:
            self.model.eval()
        
        for i in range(299):
            goal_tensor=torch.tensor(goal+one_hot)

            if usebaseline:
                all=torch.cat((goal_tensor, state),dim=0)
                velocities_tensor=self.baseline(all)
            else:
                velocities_tensor=self.model(goal_tensor, state)[0]
            
            state[:7]+=velocities_tensor

            if path_point==0 and self.close_enough(state[:7],milestone_js[0]):
                path_point=1
                one_hot=[1,0,0,0]
            elif path_point==1 and self.close_enough(state[:7],milestone_js[1]):
                path_point=2
                one_hot=[0,1,0,0]
            elif path_point==2 and self.close_enough(state[:7],milestone_js[2]):
                path_point=3
                one_hot=[0,0,1,0]           
            elif path_point==3 and self.close_enough(state[:7],milestone_js[3]):
                path_point=4
                one_hot=[0,0,0,1]   
            if use_angle:
                add=velocities_tensor
            else:
                add=state
                
            y1.append(float(add[0]))
            y2.append(float(add[1]))
            y3.append(float(add[2]))
            y4.append(float(add[3]))
            y5.append(float(add[4]))
            y6.append(float(add[5]))
            y7.append(float(add[6]))
        if return_path_point:
            return y1,y2,y3,y4,y5,y6,y7,path_point
        else:
            return y1,y2,y3,y4,y5,y6,y7

    # def get_perform(self):
    #     base_emulated = []
    #     base_offline = []
    #     base_custom = []
    #     model_offline = []
    #     model_emulated = []
    #     model_custom = []

    #     for i in range(1):
    #         # one = self.get_base_loss('emulated')
    #         # base_emulated.append(one.item())
    #         # two = self.get_base_loss('offline')
    #         # base_offline.append(two.item())
    #         # three = self.get_model_loss('emulated')
    #         # model_emulated.append(three.item())
    #         # four = self.get_model_loss('offline')
    #         # model_offline.append(four.item())
    #         five = self.calculate_cartesian_perform_model()
    #         model_custom.append(five)
    #         six = self.calculate_cartesian_perform_base()
    #         base_custom.append(six)

    #         # print('emulated base', one)
    #         # print('offline base', two)
    #         # print('emulated model', three)
    #         # print('online model', four)
    #         print('this is base perform', six)
    #         print('this is our perform', five)

    #     print()
    #     print(base_emulated)
    #     print(base_offline)
    #     print(base_custom)
    #     print(model_offline)
    #     print(model_emulated)
    #     print(model_custom)
        
    #     means_base = [np.mean(base_emulated), np.mean(base_offline), np.mean(base_custom)]
    #     means_model = [np.mean(model_offline), np.mean(model_emulated), np.mean(model_custom)]
    #     std_base = [np.std(base_emulated), np.std(base_offline), np.std(base_custom)]
    #     std_model = [np.std(model_offline), np.std(model_emulated), np.std(model_custom)]
        
    #     # Prepare data for plotting
    #     means_offline_emulated = [means_base[0], means_base[1], means_model[0], means_model[1]]
    #     std_offline_emulated = [std_base[0], std_base[1], std_model[0], std_model[1]]
    #     means_custom = [means_base[2], means_model[2]]
    #     std_custom = [std_base[2], std_model[2]]
        
    #     labels_offline_emulated = ['Base Emulated', 'Base Offline', 'Model Emulated', 'Model Offline']
    #     labels_custom = ['Base Custom', 'Model Custom']
    #     colors_offline_emulated = ['blue', 'blue', 'orange', 'orange']
    #     colors_custom = ['blue', 'orange']
        
    #     # Create first plot for Offline and Emulated metrics
    #     plt.figure(figsize=(12, 6))
    #     x_offline_emulated = range(len(means_offline_emulated))
    #     bars_offline_emulated = plt.bar(x_offline_emulated, means_offline_emulated, color=colors_offline_emulated, tick_label=labels_offline_emulated)
    #     plt.xlabel('Metrics')
    #     plt.ylabel('Values')
    #     plt.title('Offline and Emulated Metrics Comparison')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.errorbar(x_offline_emulated, means_offline_emulated, yerr=std_offline_emulated, fmt='none', ecolor='black', capsize=5, capthick=2)
    #     plt.tight_layout()
    #     plt.show()

    #     # Create second plot for Custom metrics
    #     plt.figure(figsize=(12, 6))
    #     x_custom = range(len(means_custom))
    #     bars_custom = plt.bar(x_custom, means_custom, color=colors_custom, tick_label=labels_custom)
    #     plt.xlabel('Metrics')
    #     plt.ylabel('Values')
    #     plt.title('Custom Metrics Comparison')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.errorbar(x_custom, means_custom, yerr=std_custom, fmt='none', ecolor='black', capsize=5, capthick=2)
    #     plt.tight_layout()
    #     plt.show()
    
    def get_coordinats(self,num,use_baseline):
 
        y1,y2,y3,y4,y5,y6,y7=self.get_emulated(use_baseline,num,False)
        x,y,z=[],[],[]
        for i in range(len(y1)):
            my_l=[0,0]+[y1[i]]+[y2[i]]+[y3[i]]+[y4[i]]+[y5[i]]+[y6[i]]+[y7[i]]

            p, R = self.kin.forwardkin(1, np.array(my_l))
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        return x,y,z



    def get_js_in_rad(self,num,use_baseline):
        y1,y2,y3,y4,y5,y6,y7=self.get_emulated(use_baseline,num,True)* (180.0 / math.pi)
        return y1,y2,y3,y4,y5,y6,y7


    def get_real_delta_ang(self,num,use_angle):
        y1_real,y2_real,y3_real,y4_real,y5_real,y6_real,y7_real=[],[],[],[],[],[],[]
        for j in self.dataset[num][0:]:
            if use_angle:
                delta_pos=((j[1:8]))
            else:
                delta_pos=((j[1:8]))
            y1_real.append((delta_pos[0]))
            y2_real.append((delta_pos[1]))
            y3_real.append((delta_pos[2]))
            y4_real.append((delta_pos[3]))
            y5_real.append((delta_pos[4]))
            y6_real.append((delta_pos[5]))
            y7_real.append((delta_pos[6]))
        return y1_real,y2_real,y3_real,y4_real,y5_real,y6_real,y7_real
    
    def get_real_coordinates(self, num):
        y1,y2,y3,y4,y5,y6,y7=self.get_real_delta_ang(num,False)
        x,y,z=[],[],[]
        for i in range(len(y1)):
            my_l=[0,0]+[y1[i]]+[y2[i]]+[y3[i]]+[y4[i]]+[y5[i]]+[y6[i]]+[y7[i]]
            p, R = self.kin.forwardkin(1, np.array(my_l))
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        return x,y,z

    def close_enough(self, js1,js2):
        # for i in range(7):
        #     if abs(js1[i]-js2[i])>0.05:
        #         return False
        # return True

        my_l=[0,0]
        for j in js1:
            my_l.append(float(j))
        p1, R = self.kin.forwardkin(1, np.array(my_l))
        my_l=[0,0]
        for j in js2:
            my_l.append(float(j))
        p2, R = self.kin.forwardkin(1, np.array(my_l))

        if (self.criterion2(torch.tensor(p1),torch.tensor(p2))**1/2)<0.0001:
            return True
        return False

    def get_obs_coordinates(self, num):
        elem=self.dataset[num]
        obstA=elem[1][8+self.state_size:3+8+self.state_size]
        obstB=elem[1][11+self.state_size:6+8+self.state_size]
        obstC=elem[1][14+self.state_size:9+8+self.state_size]

        return obstA,obstB,obstC

    def get_loss(self,option,use_baseline):
        all_loss=0
        for num in range(self.dataset.shape[0]):
            loss=0
            real_output=self.get_real_delta_ang(num,True)
            if option=='emulated':    
                network_output=self.get_emulated(use_baseline,num,True)
            else:
                network_output=self.get_delta_ang_offline(use_baseline,num)
            for i in range(299):
                loss+=self.criterion(torch.tensor([real_output[j][i] for j in range(7)]),torch.tensor([network_output[j][i] for j in range(7)]))
            loss/=299
            all_loss+=loss
        all_loss/=self.dataset.shape[0]
        return all_loss

    def get_changes_indexes(self,num):
        indecex=[]
        start=[0,0,0,0]
        elem=self.dataset[num]
        for i in range(299):
            goal=elem[i][self.step_size+self.state_size+self.joint_size:].tolist()
            for j in range(4):
                if goal[9+j]!=start[j]:
                    indecex.append(i)
                    start=goal[9:]
        return indecex
                

    def calculate_cartesian_perform(self,use_baseline):
        point_reached=0
        for num in range(self.dataset.shape[0]): 
            point_reached+=self.get_emulated(use_baseline,num,False,True)[7]
        return point_reached/(4*self.dataset.shape[0])   