import torch.nn as nn
import gym
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Na = env.action_space.n
        try:
            self.N = env.observation_space.n
        except:
            self.N = env.observation_space.shape[0]
        #self.linear1 = nn.Linear(self.N,128)
        #self.linear2 = nn.Linear(128,128)
        #self.linear5 = nn.Linear(128,self.Na)

        self.linear1 = nn.Linear(self.N,128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,128)
        self.linear4 = nn.Linear(128,128)
        self.linear5 = nn.Linear(128,self.Na)
        self.actv = nn.LeakyReLU()
        self.actions = torch.eye(self.Na).unsqueeze(0)
        self.states = torch.eye(self.N).unsqueeze(0)
        self.actv2 = nn.Tanh()
    def forward(self,s):
        out = self.linear1(torch.Tensor(s))
        out = self.actv2(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear5(out)
        return out
class Policy(nn.Module):
    def __init__(self,Q,epsilon = .1):
        super(Policy,self).__init__()
        self.epsilon = epsilon
        self.Q = Q
    def forward(self,s):
        amax = torch.argmax(self.Q(s)).item()
        rd = np.random.binomial(1,self.epsilon)
        action =  rd*torch.randint(0,self.Q.Na,(1,)).item() + amax*(1-rd)

        return int(action)
class update_qlearning(nn.Module):
    def __init__(self,Qvalue,Qprim,optimizerQ,gamma,Env):
        super(update_qlearning,self).__init__()
        self.Qvalue = Qvalue
        self.Qprim = Qprim
        self.optimizerQ = optimizerQ
        self.gamma = gamma
        self.Env = Env
        self.Loss = nn.MSELoss()
    def forward(self,tensordict):
        amax = self.Qvalue(tensordict["new_state"]).max(dim=1)[1]
        actions = torch.eye(self.Qvalue.Na)
        amaxT = pad_sequence([actions[a] for a in amax]).permute((1,0))
        Qprim_maxQvalue = torch.mul(self.Qprim(torch.Tensor(tensordict["new_state"])),amaxT).sum(dim=1)
        target = tensordict["reward"].squeeze() + self.gamma*Qprim_maxQvalue.squeeze()*torch.logical_not(tensordict["terminated"]).squeeze()
        target = target.detach().squeeze()
        self.optimizerQ.zero_grad()
        Qsa = torch.mul(self.Qvalue(tensordict["state"]), pad_sequence([actions[a] for a in tensordict["action"]]).permute((1,0))).sum(dim=1).squeeze()
        assert Qsa.shape==target.shape, f"verifier shape, Qsa: {Qsa.shape}, target: {target.shape}"
        loss = self.Loss(Qsa,target)
        loss.backward()
        self.optimizerQ.step()
        return loss.detach().numpy()
def swap(Qprim, Qvalue):
    tau = .9999
    for target_param, local_param in zip(Qprim.parameters(), Qvalue.parameters()):
        target_param.data.copy_(tau*local_param.data+(1-tau)*target_param.data)
class ChangeReward(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
    def step(self,action,state):
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        #if terminated and reward ==1:
        #    reward = -1
        return new_state, reward, terminated, truncated, _
class Renorm(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.mu = 0
        self.sigma = 1.0
    def fit(self,N):
        assert type(N)==int
        historic = []
        for i in range(N):
            terminated = False
            truncated = False
            state = self.reset()[0]
            while(not terminated and not truncated):
                action = self.action_space.sample()
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                state = new_state
                historic.append(state)
        self.close()
        self.mu = np.mean(historic, axis=0)
        self.sigma = np.std(historic,axis=0)
        print(f"statistics over {i} iterations")
        print("std",self.sigma)
        print("mu", self.mu)

    def step(self,action):
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        new_state = (new_state-self.mu)/self.sigma
        return new_state, reward, terminated, truncated, _
def batch(env,policy, frames_per_batch, max_frames_per_traj):
    out = {"state":[],"action":[],"new_state": [],"reward":[],"terminated": [], "truncated":[]}
    k = 0
    while k<frames_per_batch:
        state =  env.reset()[0]
        l = 0
        terminated = False
        truncated = False
        while(not terminated and not truncated and k<frames_per_batch and l<max_frames_per_traj):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action,state)
            #if new_state !=state:
            out["state"].append(state)
            out["action"].append(action)
            out["new_state"].append(new_state)
            out["reward"].append(reward)
            out["terminated"].append(terminated)
            out["truncated"].append(truncated)
            k+=1
            l+=1
            state = new_state
        env.close()
    for key in ["state", "new_state", "reward"]:
        out[key] = torch.Tensor(out[key])
    return out
