import torch
class Buffer:
    def __init__(self, maxsize = 250000):
        self.memory_state = []
        self.memory_action = []
        self.memory_newstate = []
        self.memory_reward = []
        self.memory_truncated = []
        self.memory_terminated = []
        self.maxsize = maxsize
    def store(self,sample):
        for j in range(len(sample["state"])):
            self.memory_state.append(sample["state"][j])
            self.memory_action.append(sample["action"][j])
            self.memory_newstate.append(sample["new_state"][j])
            self.memory_reward.append(sample["reward"][j])
            self.memory_truncated.append(sample["truncated"][j])
            self.memory_terminated.append(sample["terminated"][j])
        self.eviction()
        #print(self.memory_state)
    def eviction(self):
        if len(self.memory_state)>self.maxsize:
            self.memory_state = self.memory_state[-self.maxsize:]
            self.memory_action = self.memory_action[-self.maxsize:]
            self.memory_newstate = self.memory_newstate[-self.maxsize:]
            self.memory_reward = self.memory_reward[-self.maxsize:]

    def sample(self,N: int):
        assert(type(N)==int and N>0)# and N<=len(self.memory_state))
        selection = torch.randint(0,len(self.memory_state),(min(N,len(self.memory_state)),))
        state = torch.stack([self.memory_state[j] for j in selection])    
        #action = torch.stack([self.memory_action[j] for j in selection])                                                           
        action = [self.memory_action[j] for j in selection]                                                           
        newstate = torch.stack([self.memory_newstate[j] for j in selection])
        reward = torch.stack([self.memory_reward[j] for j in selection])
        terminated = torch.Tensor([self.memory_terminated[j] for j in selection])
        #renvoie un tuple de 4 tenseurs (s,a,s',r)
        sample = {"state": state,
                  "action": action,
                  "new_state": newstate,
                  "reward": reward,
                  "terminated" :terminated
                  }
        #return state, action, newstate, reward
        return sample

