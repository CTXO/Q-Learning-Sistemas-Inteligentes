import random
import time

from connection import connect, get_state_reward
import pandas as pd


s = connect(2037)


q_table = pd.read_table('resultado.txt', sep=' ', header=None)
actions = ['left', 'right', 'jump']

# Parameters
GAMMA = 0.9
ALPHA = 0.2
EPSILON = 0.05


def get_state(n_state):
    """Returns the estimated utility of each action in a given state"""
    return q_table.iloc[n_state]


def state_to_int(state):
    """Converts a state(binary) to an integer"""
    return int(state, 2)


def get_current_state_reward():
    """Retrieves the state and reward from that state that the agent is currently in"""
    data = ""
    data_recv = False
    while not data_recv:
        data = s.recv(1024).decode()
        try:
            data = eval(data)
            data_recv = True
        except:
            data_recv = False
    return data['estado'], data['recompensa']


def update_state(state, action, reward, next_state):
    state_int = state_to_int(state)
    next_state_int = state_to_int(next_state)

    state_table = get_state(state_int)
    next_state_table = get_state(next_state_int)

    q_estimate = reward + GAMMA * max(next_state_table)
    action_index = actions.index(action)
    new_q_value = state_table[action_index] + ALPHA * (q_estimate - state_table[action_index])
    print("new q value = ", new_q_value)
    print(f"Updating state {state_int} from {state_table[action_index]} to {new_q_value}")
    q_table.iloc[state_int, action_index] = new_q_value
    q_table.to_csv('resultado.txt', sep=' ', header=False, index=False)


def get_next_action(state, use_epsilon):
    state_int = state_to_int(state)
    state_table = get_state(state_int)
    if use_epsilon and random.random() <= EPSILON:
        print('random action')
        action_index = random.randint(0, 2)
    else:
        print('normal action')
        action_index = list(state_table).index(max(state_table))

    print(f"State actions utility: {list(state_table)}")
    print("action index", action_index)
    return actions[action_index]


def start_training(manual=False, n_of_plays=10000):
    state, reward = get_current_state_reward()
    for i in range(n_of_plays):
        print("Episode", i, "\n\n")

        # Train the agent manually, for didatic purposes
        if manual:
            try:
                action = input("Action: ")
            except:
                continue
            print('action', action)
            if action[0] == 'w':
                next_action = 'jump'
            elif action[0] == 'a':
                next_action = 'left'
            elif action[0] == 'd':
                next_action = 'right'
            else:
                continue
            next_state, reward = get_state_reward(s, next_action)
        else:
            next_action = get_next_action(state, use_epsilon=True)
            print("Chose Next action: ", next_action, '\n')
            next_state, reward = get_state_reward(s, next_action)
            print("Next State: ", state, "Next Reward: ", reward, '\n')
        update_state(state, next_action, reward, next_state)
        print()
        state = next_state
    print("training over")


def play_game(until_win=False, forever=False):
    state, _ = get_current_state_reward()
    continue_game = True

    while continue_game or forever:
        action = get_next_action(state, use_epsilon=False)
        next_state, reward = get_state_reward(s, action)
        int_reward = int(reward)
        continue_game = int_reward < 300 if until_win else abs(int_reward) < 100 # Checks if agent fell or won the game
        state = next_state
        time.sleep(0.1)


# start_training(manual=False, n_of_plays=5000)
play_game(until_win=True, forever=True)
