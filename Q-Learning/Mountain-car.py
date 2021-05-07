import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # b/w 0 and 1
EPISODES = 25000
#there are 3 actions in this gym -> 0.left, 1.nothing, 2.right
SHOW = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
start_decay = 1
end_decay = EPISODES // 2
epsilon_decy_value = epsilon / (end_decay - start_decay)

#initializing random q values for table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))  # ->env.action_space.n for every combination of action possible -> dimensions = 20x20x3
#print(q_table.shape, q_table.ndim)

def get_disc_state(state):
    disc_state = (state - env.observation_space.high) / discrete_os_win_size
    return tuple(disc_state.astype(np.int))

for episode in range(EPISODES):
    if episode % SHOW == 0:
        render = True
        print(EPISODES)
    else:
        render = False
    disc_state = get_disc_state(env.reset())
    #print(disc_state)
    #print(np.argmax(q_table[disc_state])) #getting max value
    done = False
    while not done:
        action = np.argmax(q_table[disc_state])
        new_state, reward, done, _ = env.step(action)
        new_disc_state = get_disc_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_disc_state])
            current_q = q_table[disc_state + (action, )] # -> getting exact q-value
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[disc_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[disc_state + (action, )] = 0

        disc_state = new_disc_state
env.close()
