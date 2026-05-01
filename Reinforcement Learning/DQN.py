# DQN - deep q networks

# q_table problem- table grows too much, not scalable
# solution - use neural networks instead 


import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

# Hyperparameters
state_size = 4
action_size = 2
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32

memory = deque(maxlen=2000)

# Model
def build_model():
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

# Store experience
def remember(s, a, r, s_next):
    memory.append((s, a, r, s_next))

# Choose action
def act(state):
    global epsilon
    if np.random.rand() < epsilon:
        return random.randrange(action_size)
    return np.argmax(model.predict(state, verbose=0)[0])

# Train
def replay():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    
    for s, a, r, s_next in batch:
        target = r + gamma * np.max(target_model.predict(s_next, verbose=0)[0])
        
        target_f = model.predict(s, verbose=0)
        target_f[0][a] = target
        
        model.fit(s, target_f, epochs=1, verbose=0)

# Training loop (dummy example)
for episode in range(50):
    state = np.random.rand(1, state_size)
    
    for step in range(10):
        action = act(state)
        
        next_state = np.random.rand(1, state_size)
        reward = 1 if np.sum(next_state) > 2 else -1
        
        remember(state, action, reward, next_state)
        
        state = next_state
        replay()
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # update target network
    target_model.set_weights(model.get_weights())

print("Training done")