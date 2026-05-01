# sarsa - state action reward state action 
# it is realistic whereas q_learning is optimistic 

# what did i actually do , we dont take max(future reward)

import numpy as np

states = 3
actions = 2

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 100

for _ in range(episodes):
    state = np.random.randint(0, states)

    # choose initial action
    if np.random.rand() < epsilon:
        action = np.random.randint(actions)
    else:
        action = np.argmax(Q[state])

    for _ in range(10):
        # transition
        next_state = np.random.randint(0, states)
        reward = 1 if next_state == 2 else -1

        # next action (IMPORTANT difference)
        if np.random.rand() < epsilon:
            next_action = np.random.randint(actions)
        else:
            next_action = np.argmax(Q[next_state])

        # SARSA update
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action

print("Q-table:")
print(Q)