# Double Q-Learning Implementation for CarPole environnement 

I implement double Q learning with a replay buffer.

* File arg.json contains the hyperparameters: learning rate $\alpha$, exploration rate $\epsilon$, the number of episodes, the number of iterations in the updating rule inner loop $K$
* File buffer.py contains the replay buffer implementatation
* File main.py contains the double q learning inner loop
* File utils.py contains the implementation of the class for the state action value fonction Q, the policy, which is only the action which maximizes the Q function, and updateclass for updating the Q function
For each episode we visualise the discounted sum of the rewards

![Alt text](image/retour2.png)


