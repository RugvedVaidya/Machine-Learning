### REINFORCEMENT LEARNING

# no labels
# no fixed dataset 
# learns from interecation + reward (feedback)
# eg : in a game of chess - you play moves - you win or lose - you improve accordingly 

### core components : 
#     1. agent - model 
#     2. Environment - system 
#     3. state(s) - current situation (eg : position in chess)
#     4. action - what agent can do 
#     5. reward(R) - feedback after action (eg : if good move you go + in chess, if bad move you go - chess)
#     6. policy - strategy of agent 
    
### main loop -> state - action - reward - new state - repeat

### objective -> maximize the total reward over time 

# Gt = R(t+1) + yR(t+2) + y^2R(t+3) + ... 
# Gt - total future reward 
# y - discount factor (0 to 1) if close to 1 care about long term reward else care about immediate reward

### exploration vs exploitation 
# exploration - try new actions
# exploitation - use what you already know
# eg : always playing e4 with white - exploitation
#     playing a4 with white - exploration
    
# types of RL :
#     1. value based (Q- learning)
#     2. policy based 
#     3. model based vs model free 

### Q- learning 
# learns a function: 
#     "how good is it to take action A in state S ? this is called Q-Value"
# higher value - better action 
# agent always prefers higher Q 

# eg : how good is that move in that current position, higher the + reward, more good the move 

# learning process : 
#     1. start in a state 
#     2. take an action 
#     3. get reward 
#     4. move to next state 
#     5. update Q value 
#     6. repeat 

### formula : Q(s,a)=Q(s,a)+α[r+γ*max(a′)*​Q(s′,a′)−Q(s,a)]
    # Q(s,a) - current state 
    # Q(s′,a′) - best future reward 
    # y(gamma) - discount factor 
    # r - reward 
    # α - learning rate
    
    # old value = new value + correction 
    # correctino = actual reward + future best - current estimate 

### eg : goal - reach destinatino - +10 , wrong move - -1
    #     agent learns : which path gives highest reward 
    
    # exploration vs exploitation :
    #     e- greedy strategy (e- epsilon)
    #     with e - explore 
    #     with 1-e - exploit
        
    # algo : 
    #     1. initialize q table (all zero values)
    #     2. for each episode 
    #         start at initial staticmethod
    #         choose action 
    #         take action 
    #         get reward + next state 
    #         update q-table 
    #     3. repeat many time 
        
    # if enviroment is random - learning is meaningless
    # if enviroment has structure - learning is meaningful
    
import numpy as np

#environment 
states = 3
actions = 2

#q-table
q_table = np.zeros((states, actions))

#parameters 
alpha = 0.1
gamma = 0.9
epsilon = 0.2 
episodes = 100

for _ in range(episodes):
    state = np.random.randint(0, states)
    
    for _ in range(10):
        #epsilon greedy
        if np.random.rand() < epsilon :
            action = np.random.randint(actions)
        else:
            action = np.argmax(q_table[state])
            
        # dummy transaction
        next_state = np.random.randint(0, states)
        if state == 0 and action == 1:
            next_state = 1
        elif state == 1 and action == 1:
            next_state = 2
        else:
            next_state = state
        reward =1 if next_state == 2 else -1
        
        # q-table update 
        q_table[state, action] += alpha * (reward + gamma *np.max(q_table[next_state]) - q_table[state, action])
        
        state = next_state
        
print("q_table : ")
print(q_table)