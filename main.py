import pygame
import random
from snake import Snake
import numpy as np

# Initialization

# fixed seed for deterministic behavior:
np.random.seed(100)
random.seed(100)

# dimensions
feature_size = 6
action_size = 4


# learnable weight initialization
# these are the global weights we update during the learning
betas = np.random.rand(feature_size, action_size)

# number of transitions the agent performs Q learning
len_epoch = 20000

# hyperparameters 
temp = 100.0
lr = 0.2 
gamma = 0.5 

# action translator. The snake environment expects string keywords, but for the RL agent it is simpler to output
# indices between 0 and 3.
actions = {
    0: "left",
    1: "up",
    2: "right",
    3: "down"
}

def approximate_q_values(phi_s):
    """
    phi_s: Feature vector with shape (feature_size,). Extracted features from a state s.
    
    return: Vector containing the 4 Q-values Q(s,a) for every possible action a.
    """
    approximated_q_value = np.dot(phi_s, betas)
    return approximated_q_value

def softmax(x):
    """
    x: 1-dimensional numpy array.
    returns: softmax(x), 1-dimensional numpy.
    """
    return np.exp(x) / np.sum(np.exp(x))

def select_action(phi_s, sample=True):
    """
    phi_s: Feature vector with shape (feature_size,). Extracted features from a state s.
    sample: Boolean flag. If true, sample an action based on the Soft-Max policy, otherwise return the action with
            the highest Q-value.
    
    return: Integer action index a_idx (in [0,1,2,3]) of the selected action.
    """
    qs = approximate_q_values(phi_s)
    if sample:
        selection_probabilities = softmax(qs / temp)

        result = np.random.multinomial(1, selection_probabilities)
        a_idx = np.argmax(result)
    else:
        a_idx = np.argmax(qs)
        
    return a_idx

def compute_delta(phi_s, a_idx, r, phi_new_s, is_terminal):
    """
    phi_s: Feature vector of state s with shape (feature_size,).
    a_idx: action index a_idx (in [0,1,2,3]) of the action selected in state s.
    r: reward r(s,a) of the action with index a_idx in state s (float).
    phi_new_s: Feature vector of state new_s with shape (feature_size,). 
               new_s is the state after selection action a_idx in state s.
    is_terminal: boolean which indicates if s is a terminal state.
    
    return: td_error delta of type float
    """
    current_q = approximate_q_values(phi_s)[a_idx]
    if is_terminal:
        delta = r - current_q
    else:
        delta = r + gamma * np.max(approximate_q_values(phi_new_s)) - current_q
    return delta

def compute_d_qsa_d_beta(phi_s, a_idx):
    """
    phi_s: Feature vector of state s with shape (feature_size,).
    a_idx: action index a_idx (in [0,1,2,3]) of the action selected in state s.
    
    return: Derivative of the q_value Q(s, a) wrt. betas. It has shape (feature_size, action_size)
    """
    d_qsa_d_beta = np.zeros((feature_size, action_size))
    d_qsa_d_beta[:, a_idx] = phi_s
    return d_qsa_d_beta

def update_betas(phi_s, a_idx, r, phi_new_s, is_terminal):
    """
    phi_s: Feature vector of state s with shape (feature_size,).
    a_idx: action index a_idx (in [0,1,2,3]) of the action selected in state s.
    r: reward r(s,a) of the action with index a_idx in state s (float).
    phi_new_s: Feature vector of state new_s with shape (feature_size,). 
               new_s is the state after selection action a_idx in state s.
    is_terminal: boolean which indicates if s is a terminal state.
    
    return: None, but you have to change the value of the global betas
    """
    # we want to update the global variable beta, hence the global "import"
    global betas
    delta = compute_delta(phi_s, a_idx, r, phi_new_s, is_terminal)
    d_qsa_d_beta = compute_d_qsa_d_beta(phi_s, a_idx)
    betas = betas + lr * delta * d_qsa_d_beta

# helper function to test the current policy
def test_policy(num_games=5):
    av_score = 0
    g = 0
    snake = Snake(FRAMESPEED=50000)
    while g < num_games:
        s = snake.get_feature_representation()
        a_idx = select_action(s, sample=False)
        a = actions[a_idx]
        is_terminal = snake.step(a)
        if is_terminal:
            av_score += snake.last_score
            g += 1
    pygame.quit()

    av_score /= num_games
    print(f"Average score: {av_score}")

# helper function to simulate one game with normal speed
def play_single_game(framespeed=20):
    snake = Snake(FRAMESPEED=framespeed)
    while True:
        s = snake.get_feature_representation()
        a_idx = select_action(s, sample=False)
        is_terminal = snake.step(actions[a_idx], init_new_game_after_terminal=False)
        if is_terminal:
            print(f"Total Score: {snake.last_score}")
            break

snake = Snake(FRAMESPEED=50000)

for i in range(len_epoch):
    # 1. get the feature representation of the current state by calling snake.get_feature_representation()
    phi_s = snake.get_feature_representation()
    # 2. Select the action using the epsilon-greedy exploration strategy
    a_idx = select_action(phi_s, sample=True)
    # 3. Get the action string which is needed for the snake environment. (already implemented)
    a = actions[a_idx]
    # 4. Ask for the current reward (already impelmented)
    r = snake.get_reward(a)
    # 5. Perform one step. This returns the boolean is_terminal if the snake died during this step.
    # (already implemented)
    is_terminal = snake.step(a)
    # 6. Get the feature representation of the updated state by calling snake.get_feature_representation()
    phi_new_s = snake.get_feature_representation()
    # 7. Update betas by calling the update_betas(...) method
    update_betas(phi_s, a_idx, r, phi_new_s, is_terminal)
        
    # To see how well the current policy works we test it every 5000 updates
    if i % 5000 == 0:   
        pygame.quit()
        test_policy()
        snake = Snake(FRAMESPEED=50000)
    
    # update hyperparameters
    temp *= 0.999
    temp = max(temp, 0.1)
    #print(epsilon)
    lr *= 0.9999

test_policy()
pygame.quit()

play_single_game(framespeed=50)
pygame.quit()   