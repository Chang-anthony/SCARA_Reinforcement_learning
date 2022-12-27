import gym 
import numpy as np
from plot import plot_learning_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import My_ReplayBuffer,DataBuffer
import os
import time
path = os.path.dirname(__file__)+'/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu ")
torch.autograd.set_detect_anomaly(True)


################# hyper parameter #####################
env = gym.make("Pendulum-v1")
N_GAME = 6000
EP_LEN = 128
N_EPOCH = 10
A_LR = 2e-5
C_LR = 2e-5
tau = 0.005
gamma = 0.95
gae_lamda = 0.9
epslion = 0.2
min_batch = 64
#######################################################

class Actor_Continous(nn.Module):
    def __init__(self,n_state,n_action,qhigh,qdhigh=None,LR=None,std_min=1e-4,std_max=1,fc1=64,fc2=128,fc3=256,fc4=256,fc5=256):
        super(Actor_Continous,self).__init__()
        self.h1 = nn.Linear(n_state,fc1).to(device)
        self.h2 = nn.Linear(fc1,fc2).to(device)
        self.h3 = nn.Linear(fc2,fc3).to(device)
        self.h4 = nn.Linear(fc3,fc4).to(device)
        self.h5 = nn.Linear(fc4,fc5).to(device)
        
    
        
        self.n_state = n_state
        self.n_action = n_action
        self.std_min = std_min
        self.std_max = std_max
        
        self.mu = nn.Linear(fc5,n_action).to(device)
        self.sigma = nn.Linear(fc5,n_action).to(device)
        self.qhigh = qhigh
        self.qdhigh = qdhigh 
        if LR is not None:
            self.optim = torch.optim.Adam(self.parameters(),lr=LR)
           
    def forward(self,state):
        x = F.gelu(self.h1(state))
        x = F.gelu(self.h2(x))
        x = F.gelu(self.h3(x))
        x = F.gelu(self.h4(x))
        x = F.gelu(self.h5(x))
        
        mu = torch.tanh(self.mu(x)) 
        if self.qdhigh is not None:
            muq = mu[:,0:2] * self.qhigh
            muqd = mu[:,2:self.n_action] * self.qdhigh
            mu = torch.cat([muq,muqd],dim=1) 
        else:
            mu = mu * self.qhigh
        
        sigma = F.softplus(self.sigma(x))
        sigma = torch.clamp(sigma,min=self.std_min,max=self.std_max)
        
        return mu,sigma
    
class Critic(nn.Module):
    def __init__(self,n_state,LR=None,fc1=64,fc2=128,fc3=256):
        super(Critic,self).__init__()
        
        self.h1 = nn.Linear(n_state,fc1).to(device)
        self.h2 = nn.Linear(fc1,fc2).to(device)
        self.h3 = nn.Linear(fc2,fc3).to(device)
        
        self.v = nn.Linear(fc3,1).to(device)
        if LR is not None:
            self.optim = torch.optim.Adam(self.parameters(),lr=LR)
    
    def forward(self,state):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        v = self.v(x)
        
        return v
    
class PPO(object):
        def __init__(self,n_state,
                    n_action,
                    action_q_low,
                    action_q_high,
                    action_qd_low,
                    action_qd_high,
                    std_min = 1e-4,
                    std_max = 1,
                    entropy_cofe = 0.01,
                    A_LR = 0.001,
                    C_LR = 0.002,
                    epslion = 0.2,
                    tau = 0.005,
                    gamma = 0.99,
                    gae_lamda = 0.95,
                    norm_adv = False,
                    max_size = 100000,
                    n_epoch = 5,
                    fc1 = 32,
                    fc2 = 64,
                    fc3 = 128,
                    fc4 = 256,
                    fc5 = 256,
                    save_name = 'v1.pth',
                    load_name = '.pth',
                    path = os.path.dirname(__file__)+'/PPO',
                    batch_size = 64,
                    ):
            
            self.pi = Actor_Continous(n_state,n_action,action_q_high,action_qd_high,A_LR,std_min=std_min,std_max=std_max).to(device)
            self.old_pi = Actor_Continous(n_state,n_action,action_q_high,action_qd_high).to(device)
            
            self.v = Critic(n_state,C_LR).to(device)
            self.old_v = Critic(n_state).to(device)


            #parameter
            self.entropy_cofe = entropy_cofe
            self.n_state = n_state
            self.n_action = n_action
            self.qlow = action_q_low
            self.qhigh = action_q_high
            self.qdlow = action_qd_low
            self.qdhigh = action_qd_high
            self.std_min = std_min
            self.std_max = std_max
            self.A_LR = A_LR
            self.C_LR = C_LR
            self.epslion = epslion
            self.tau = tau
            self.gamma = gamma
            self.gae_lamba = gae_lamda
            self.norm_adv = norm_adv
            self.n_epoch = n_epoch
            self.memory = My_ReplayBuffer(max_size)
            self.path = path
            self.save_name = save_name
            self.load_name = load_name
            self.batch_size = batch_size
            self.buffer = DataBuffer()
            self.load_model()

          
        def __soft_update(self,old_net,target_net):
            for  target,net in zip(target_net.parameters(),old_net.parameters()):
                target.data.copy_(target.data * (1-self.tau) + net.data * tau)
        
        def store_transition(self,s,a,r,s_,done):
            self.memory.store_transition(s,a,r,s_,done)
            # self.buffer.store_transition(s,a,r,s_,done)
            
        def choose_action(self,state):
            state = torch.tensor([state],dtype=torch.float32).to(device)
            with torch.no_grad():
                mu,sigma = self.pi(state)
                mu = mu 
                pi = torch.distributions.normal.Normal(loc=mu,scale=sigma)
                action = pi.sample()
                action = action.cpu().detach().numpy()
                
                action = action.reshape(-1)
                if self.qdhigh is not None:
                    actionq = np.clip(action[0:2],self.qlow,self.qhigh)
                    actionqd = np.clip(action[2:self.n_action],self.qdlow,self.qdhigh)
                    action = np.concatenate((actionq,actionqd))
                # action = np.clip(action,self.low,self.high).reshape(-1)
                else:
                    action = np.clip(action,self.qlow,self.qhigh)
                    
            return action
        
        
        def clculate_advantage(self,td_error):
            advantage = []
            adv = 0.0
            #反著計算這樣計算的結果剛好最後一個的狀態的 td_error會乘上最多的衰減率
            for td in td_error[::-1]:
                adv = self.gamma * self.gae_lamba * adv + td
                advantage.append(adv)
            advantage.reverse()
            
            adv = torch.tensor([advantage],dtype=torch.float32).reshape(-1,1).to(device)
            if self.norm_adv:
                #1e-5 is avoid to dvide by 0 ,sometime helpful
                adv = (adv - adv.mean())/(adv.std() + 1e-5)
            return adv
        
        def learn(self):
            
            if self.memory.size() < self.batch_size:
                return

            for i in range(self.n_epoch):
                s,a,r,s_,done = self.memory.sample(self.batch_size)
                # s,a,r,s_,done = self.memory.sample_no_randonm()
                # s,a,r,s_,done = self.buffer.sample()
                # if len(r) < self.batch_size:
                #     return
                
                state = torch.tensor(s,dtype=torch.float32).to(device)
                action = torch.tensor(a,dtype=torch.float32).reshape(-1,self.n_action).to(device)
                reward = torch.tensor(r,dtype=torch.float32).reshape(-1,1).to(device)
                next_state = torch.tensor(s_,dtype=torch.float32).to(device)
                dones = torch.tensor(done,dtype=torch.float32).reshape(-1,1).to(device)
                
                # reward = (reward - reward.mean())/(reward.std()+1e-5)
                with torch.no_grad():
                    '''
                        td_target
                    '''
                    # old_v = torch.squeeze(self.old_v(next_state),1)
                    # td_target = reward + self.gamma * old_v * (1 - dones)
                    # td_target = reward + self.gamma * self.old_v(next_state) * (1 - dones)
                    td_target = reward + self.gamma * self.v(next_state) * (1 - dones)
                    '''
                        advantage
                    '''
                    # new_v = torch.squeeze(self.v(next_state),1)
                    # v = torch.squeeze(self.v(state),1)
                    # td_error = reward + self.gamma * new_v * (1 - dones) - v
                    td_error = reward + self.gamma * self.v(next_state) * (1 - dones) - self.v(state)
                    td_error = td_error.cpu().detach().numpy()
                    advantage = self.clculate_advantage(td_error) #[batch_size,1] 
                    '''
                        old_pi 
                    '''
                    mu,sigma = self.old_pi(state)
                    old_dis = torch.distributions.Normal(mu,sigma)
                    old_log_prob = old_dis.log_prob(action).detach()
                '''
                    pi loss
                '''
                mu,sigma = self.pi(state)
                new_dis = torch.distributions.Normal(mu,sigma)
                #如果batch size = 64 以下的計算結果都要為 [batch_size,1]
                log_prob = new_dis.log_prob(action)
                entropy = new_dis.entropy()
                ratio = torch.exp(log_prob - old_log_prob)
                surr = ratio * advantage
                x1 = torch.clamp(ratio,1-self.epslion,1+self.epslion) * advantage
                

                loss_pi = -torch.min(surr,x1).mean()-self.entropy_cofe * torch.mean(entropy) + 0.5 * F.mse_loss(self.v(state),td_target.detach())
                # loss_v = loss.copy_(loss)
                loss_v = 0.5 * F.mse_loss(self.v(state),td_target.detach())
                # pi_loss = -torch.min(surr,x1).mean()
                # pi_loss = pi_loss - torch.mean(self.entropy_cofe * entropy)
                self.pi.optim.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(),0.5)
                self.pi.optim.step()
                '''
                    v_loss
                '''
                # loss_v = torch.mean(F.mse_loss(torch.squeeze(self.v(state),1),td_target))
                # loss_v = torch.mean(F.mse_loss(self.v(state),td_target.detach()))
                self.v.optim.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.v.parameters(),0.5)
                self.v.optim.step()
                
            self.memory.clear()
            # self.buffer.clear()
            self.old_pi.load_state_dict(self.pi.state_dict())
            # self.__soft_update(self.v,self.old_v)
            # self.old_v.load_state_dict(self.v.state_dict())
                
        def save_model(self):
            torch.save(self.pi.state_dict(),self.path+'pi'+self.save_name)
            torch.save(self.old_pi.state_dict(),self.path+'old_pi'+self.save_name)
            torch.save(self.v.state_dict(),self.path+'v'+self.save_name)
            torch.save(self.old_v.state_dict(),self.path+'target_v'+self.save_name)
            print("-------------------- save model -----------------")
            
        def load_model(self):
            try:
                self.pi.load_state_dict(torch.load(self.path+'pi'+self.load_name))
                self.old_pi.load_state_dict(torch.load(self.path+'old_pi'+self.load_name))
                self.v.load_state_dict(torch.load(self.path+'v'+self.load_name))
                self.old_v.load_state_dict(torch.load(self.path+'target_v'+self.load_name))
                print("--------------------- load model --------------------")
            except:
                pass
            
if __name__ == "__main__":
    figure_file = os.path.dirname(__file__)+'/plots/cartpole.png'
    action_dim = env.action_space.shape[0]
    high = env.action_space.high[0] 
    low = env.action_space.low[0]
    state_dim = env.observation_space.shape[0]
    best_score = env.reward_range[0]
    score_history = []
    avg_score = 0
    agent = PPO(state_dim,action_dim,low,high,A_LR,C_LR,epslion,tau,gamma,gae_lamda,norm_adv=False,n_epoch=N_EPOCH,path=path)
    max_rewards = -1000000
    for _ in range(N_GAME):
        s = env.reset()
        start = True
        score = 0
        if _ % 10 == 0:
            t0 = time.time()
        while start:
            for i in range(EP_LEN):
                # env.render()
                a = agent.choose_action(s)
                s_,r,done,info = env.step(a)
                score += r
                agent.store_transition(s,a,(r+8.1)/8.1,s_,done)
                if done:
                    start = False
                    break
                s = s_
            agent.learn()
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if _ % 10 == 0:
            t = (time.time() - t0)  
            print(_, 'score for 10 run %.3f ' % score, 'time for 10 run %.3f' % t )
        if score > max_rewards:
            max_rewards = score
            print('max_rewards %.3f' % max_rewards)
            agent.save_model()
        
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x,score_history,figure_file)