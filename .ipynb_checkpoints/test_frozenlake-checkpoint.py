import gym




def run():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode ="human")
    state =  env.reset()[0]
    terminated = False
    truncated = False #true

    while(not terminated and not truncated):
        #actions: 0,1,2,3
        #states: 0 is left corner, 63 is the bottom right corner
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state
    env.close()


if __name__=="__main__":
   run() 
