import warnings
from torch.nn.utils.rnn import pad_sequence
warnings.filterwarnings("ignore")
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.distributions import Categorical
import torch.optim as optim
from torch.optim.lr_scheduler import MultiplicativeLR
from utils import Q,Policy,  swap,update_qlearning,ChangeReward,batch, Renorm
import gym
import json
import numpy as np
from buffer import Buffer
from torch.nn.utils import clip_grad_norm_
def test(policy,gamma, sigma = 1,mu = 0):
    retour = 0
    env = gym.make("CartPole-v1")#, render_mode = "human")
    env = Renorm(env)
    env.sigma = sigma
    env = ChangeReward(env)
    epsilon = policy.epsilon
    policy.epsilon = 0
    terminated = False
    truncated = False
    state = env.reset()[0]
    k=0
    while(not terminated and not truncated):
        action = policy(state)
        new_state, reward, terminated, truncated, _ = env.step(action,state)
        state = new_state
        retour+=reward*gamma**k
        k+=1
    #print("k",k)
    env.close()
    policy.epsilon = epsilon
    return retour
def output(policy, env, list_loss, list_retour, list_episodes,list_epsilon):
    list_retour.append(test(policy,gamma, env.sigma,env.mu))
    list_loss.append(loss.item())
    list_episodes.append(i)
    list_epsilon.append(policy.epsilon)
    
    plt.figure()
    plt.semilogy(list_episodes,list_loss)
    plt.xlabel("episodes")
    plt.title("Loss")
    plt.savefig("plot")
    plt.figure()
    plt.plot(list_episodes, list_retour)
    plt.xlabel("episodes")
    plt.grid()
    plt.savefig("retour")
    plt.figure()
    plt.title("epsilon")
    plt.plot(list_episodes, list_epsilon)
    plt.savefig("epsilon")
if __name__=="__main__":
    device = torch.device("cpu")
    with open("arg.json","r") as f:
        data = json.load(f)
    train = data["train"]
    test_mode = data["test_mode"]
    start = data["start"]
    epsilon = data["epsilon"]
    gamma = data["gamma"]
    lr = data["lr"]
    loadpath = data["loadpath"]
    loadopt = data["loadopt"]
    K = data["K"]
    pathImage = data["pathImage"]
    n_episodes = data["n_episodes"]

    sub_batch_size = data["sub_batch_size"]
    memory_size = data["memory_size"]
    N = data["N"]

    env = gym.make("CartPole-v1")
    #env = gym.make("CartPole-v1")
    env = Renorm(env)

    #env.fit(1000)
    env = ChangeReward(env)
    buffer = Buffer(memory_size)

    Qvalue = Q(env)
    Qprim = Q(env)
    optimizerQ = optim.Adam(Qvalue.parameters(), lr = lr)
    clip_grad_norm_(Qvalue.parameters(), 1.0)
    lmbda = lambda epoch: 1.0
    scheduler = MultiplicativeLR(optimizerQ, lr_lambda=lmbda)
    
    if start>0:
        Qvalue.load_state_dict(torch.load(os.path.join(loadpath,f"q_load_{start}.pt"), weights_only=True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_load_{start}.pt"), weights_only=True))
    policy = Policy(Qvalue, epsilon = epsilon)
    update_class = update_qlearning(Qvalue,Qprim,optimizerQ,gamma,env)

    
    list_loss = []
    list_retour = []
    list_episodes = []
    list_epsilon = []
    coef = .999
    n = 0
    if train:
        for i in tqdm(range(start,n_episodes)):
            truncated = False
            terminated = False
            state = env.reset()[0]
            retour = 0
            j =0
            k=0
            while(not terminated and not truncated and k<1000):
                if n%N==0:
                    swap(Qprim, Qvalue)
                action = policy(state)
                new_state, reward, terminated, truncated, _ = env.step(action,state)
                buffer.store({"state":      [torch.Tensor(state)],
                              "action":     [action],
                              "new_state":  [torch.Tensor(new_state)],
                              "reward":     [torch.Tensor([reward])],
                              "truncated":  [truncated],
                              "terminated": [terminated]})
                state = new_state
                for k in range(K):
                    subdata = buffer.sample(sub_batch_size)
                    loss = update_class(subdata)
                    scheduler.step()
                retour+=reward*(gamma**j)
                n+=1
                j+=1
                k+=1
            if policy.epsilon>0.01:
                policy.epsilon *=coef
            if i%10==0:
                output(policy, env, list_loss, list_retour, list_episodes, list_epsilon)
            if i%100==0 and i>0:
                torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{i}.pt"))
                torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{i}.pt"))
    test(policy,gamma)
