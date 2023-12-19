# DQN
import numpy as np
import math
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# 2048
import curses
from random import randrange, choice # generate and place new tile
from collections import defaultdict

import csv

# DQN class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1

    def forward(self, x):
        self.z1 = np.dot(x, self.w1)
        self.a1 = np.maximum(self.z1, 0)  # ReLU activation
        self.z2 = np.dot(self.a1, self.w2)
        return self.z2

    def backward(self, x, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error

        hidden_error = np.dot(output_delta, self.w2.T)
        hidden_delta = hidden_error * (self.z1 > 0)  # Derivative of ReLU

        self.w2 += np.dot(self.a1.T, output_delta) * learning_rate
        self.w1 += np.dot(x.T, hidden_delta) * learning_rate

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, gamma=0.95, epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.model = NeuralNetwork(state_size, hidden_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.forward(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.forward(next_state)[0])
            target_f = self.model.forward(state)
            target_f[0][action] = target
            self.model.backward(state, target_f, self.model.forward(state), self.learning_rate)




# -----------------------------------------------------------
# 2048 game, the game itself we use another code from github
letter_codes = [ord(ch) for ch in 'WASDRQwasdrq']
actions = ['Up', 'Left', 'Down', 'Right', 'Restart', 'Exit']
actions_dict = dict(zip(letter_codes, actions * 2))

def get_user_action(keyboard):    
    char = "N"
    while char not in actions_dict:    
        char = keyboard.getch()
    return actions_dict[char]

def transpose(field):
    return [list(row) for row in zip(*field)]

def invert(field):
    return [row[::-1] for row in field]

class GameField(object):
    def __init__(self, height=4, width=4, win=2048):
        self.height = height
        self.width = width
        self.win_value = win
        self.score = 0
        self.highscore = 0
        self.reset()

    def reset(self):
        if self.score > self.highscore:
            self.highscore = self.score
        self.score = 0
        self.field = [[0 for i in range(self.width)] for j in range(self.height)]
        self.spawn()
        self.spawn()

    def move(self, direction):
        def move_row_left(row):
            def tighten(row): # squeese non-zero elements together
                new_row = [i for i in row if i != 0]
                new_row += [0 for i in range(len(row) - len(new_row))]
                return new_row

            def merge(row):
                pair = False
                new_row = []
                for i in range(len(row)):
                    if pair:
                        new_row.append(2 * row[i])
                        self.score += 2 * row[i]
                        pair = False
                    else:
                        if i + 1 < len(row) and row[i] == row[i + 1]:
                            pair = True
                            new_row.append(0)
                        else:
                            new_row.append(row[i])
                assert len(new_row) == len(row)
                return new_row
            return tighten(merge(tighten(row)))

        moves = {}
        moves['Left']  = lambda field:                              \
                [move_row_left(row) for row in field]
        moves['Right'] = lambda field:                              \
                invert(moves['Left'](invert(field)))
        moves['Up']    = lambda field:                              \
                transpose(moves['Left'](transpose(field)))
        moves['Down']  = lambda field:                              \
                transpose(moves['Right'](transpose(field)))

        if direction in moves:
            if self.move_is_possible(direction):
                self.field = moves[direction](self.field)
                self.spawn()
                return True
            else:
                return False

    def is_win(self):
        return any(any(i >= self.win_value for i in row) for row in self.field)

    def is_gameover(self):
        return not any(self.move_is_possible(move) for move in actions)

    def draw(self, screen):
        help_string1 = '(W)Up (S)Down (A)Left (D)Right'
        help_string2 = '     (R)Restart (Q)Exit'
        gameover_string = '           GAME OVER'
        win_string = '          YOU WIN!'
        def cast(string):
            screen.addstr(string + '\n')

        def draw_hor_separator():
            line = '+' + ('+------' * self.width + '+')[1:]
            separator = defaultdict(lambda: line)
            if not hasattr(draw_hor_separator, "counter"):
                draw_hor_separator.counter = 0
            cast(separator[draw_hor_separator.counter])
            draw_hor_separator.counter += 1

        def draw_row(row):
            cast(''.join('|{: ^5} '.format(num) if num > 0 else '|      ' for num in row) + '|')

        screen.clear()
        cast('SCORE: ' + str(self.score))
        if 0 != self.highscore:
            cast('HIGHSCORE: ' + str(self.highscore))
        for row in self.field:
            draw_hor_separator()
            draw_row(row)
        draw_hor_separator()
        if self.is_win():
            cast(win_string)
        else:
            if self.is_gameover():
                cast(gameover_string)
            else:
                cast(help_string1)
        cast(help_string2)

    def spawn(self):
        new_element = 4 if randrange(100) > 89 else 2
        (i,j) = choice([(i,j) for i in range(self.width) for j in range(self.height) if self.field[i][j] == 0])
        self.field[i][j] = new_element

    def move_is_possible(self, direction):
        def row_is_left_movable(row): 
            def change(i): # true if there'll be change in i-th tile
                if row[i] == 0 and row[i + 1] != 0: # Move
                    return True
                if row[i] != 0 and row[i + 1] == row[i]: # Merge
                    return True
                return False
            return any(change(i) for i in range(len(row) - 1))

        check = {}
        check['Left']  = lambda field:                              \
                any(row_is_left_movable(row) for row in field)

        check['Right'] = lambda field:                              \
                 check['Left'](invert(field))

        check['Up']    = lambda field:                              \
                check['Left'](transpose(field))

        check['Down']  = lambda field:                              \
                check['Right'](transpose(field))

        if direction in check:
            return check[direction](self.field)
        else:
            return False

# we don't actually need to play the game by our self, so just command this part
# play the game

# def main(stdscr):
#     def init():
#         #重置游戏棋盘
#         game_field.reset()
#         return 'Game'

#     def not_game(state):
#         #画出 GameOver 或者 Win 的界面
#         game_field.draw(stdscr)
#         #读取用户输入得到action，判断是重启游戏还是结束游戏
#         action = get_user_action(stdscr)
#         responses = defaultdict(lambda: state) #默认是当前状态，没有行为就会一直在当前界面循环
#         responses['Restart'], responses['Exit'] = 'Init', 'Exit' #对应不同的行为转换到不同的状态
#         return responses[action]

#     def game():
#         #画出当前棋盘状态
#         game_field.draw(stdscr)
#         #读取用户输入得到action
#         action = get_user_action(stdscr)

#         if action == 'Restart':
#             return 'Init'
#         if action == 'Exit':
#             return 'Exit'
#         if game_field.move(action): # move successful
#             if game_field.is_win():
#                 return 'Win'
#             if game_field.is_gameover():
#                 return 'Gameover'
#         return 'Game'


#     state_actions = {
#             'Init': init,
#             'Win': lambda: not_game('Win'),
#             'Gameover': lambda: not_game('Gameover'),
#             'Game': game
#         }

#     curses.use_default_colors()

#     # 设置终结状态最大数值为 2048
#     game_field = GameField(win=2048)


#     state = 'Init'

#     #状态机开始循环
#     while state != 'Exit':
#         state = state_actions[state]()
# curses.wrapper(main)

import pickle

def save_model(agent, episode):
    model_data = {
        "weights": {
            "w1": agent.model.w1,
            "w2": agent.model.w2
        },
        "epsilon": agent.epsilon
    }
    with open(f".\\agents\\2048_model_episode_{episode}.pkl", "wb") as file:
        pickle.dump(model_data, file)
    print(f"Model saved to 2048_model_episode_{episode}.pkl")


def load_model(filename, state_size, action_size):
    with open(filename, "rb") as file:
        model_data = pickle.load(file)

    agent = DQNAgent(state_size, action_size)
    agent.model.w1 = model_data["weights"]["w1"]
    agent.model.w2 = model_data["weights"]["w2"]
    agent.epsilon = model_data["epsilon"]

    return agent


def train_and_play(stdscr, episodes=100, batch_size=128):
    state_size = 16  # the game field is 4x4
    action_size = 4  # 4 actions: up, down, left, right
    agent = DQNAgent(state_size, action_size)

    with open ("2048_DQN_self_model.csv", "w", newline='') as file:
        writer = csv.writer(file)
        data = ["episode", "score", "average_reward", "epsilon"]
        writer.writerow(data)
        for episode in range(episodes):
            # initialize the game field
            game_field = GameField()
            state = np.array(game_field.field).flatten().reshape(1, 16)
            done = False
            average_reward = 0
            num_moves = 0
            # average_reward = average_reward / num_moves
            score = 0
            data = [episode, score, agent.epsilon]

            while not done:
                num_moves += 1

                # choose an action
                action = agent.act(state)

                # apply the action to the game field
                moved = game_field.move(actions[action])
                next_state = np.array(game_field.field).flatten().reshape(1, 16)
                # reward = game_field.score - score if moved else -1  
                reward = game_field.score - score if moved else 0  
                reward = math.log2(reward) if reward > 0 else reward
                
                average_reward += reward
                score = game_field.score
                done = game_field.is_gameover()
                reward = -100 if done else reward

                # store to memory
                agent.remember(state, action, reward, next_state, done)

                # update state
                state = next_state

                # learn
                agent.replay(batch_size)

                # print the game field
                game_field.draw(stdscr)
                stdscr.refresh()

            # update epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            # save the model
            if episode % 1 == 0:
                save_model(agent, episode)

            # average_reward = average_reward / num_moves
            data = [episode, score, average_reward, agent.epsilon]
            writer.writerow(data)


def test_and_play(stdscr, filename, episodes=1):
    # load teh model
    state_size = 16  
    action_size = 4  
    agent = load_model(filename, state_size, action_size)

    for episode in range(episodes):
        game_field = GameField()
        state = np.array(game_field.field).flatten().reshape(1, 16)
        done = False

        while not done:
            # let the agent decide
            action = agent.act(state)
            moved = game_field.move(actions[action])
            next_state = np.array(game_field.field).flatten().reshape(1, 16)
            done = game_field.is_gameover()
            state = next_state

            # draw
            game_field.draw(stdscr)
            stdscr.refresh()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
# test
# filename = './agents/2048_model_episode_2824.pkl'  # file name
# curses.wrapper(lambda stdscr: test_and_play(stdscr, filename))


# train
curses.wrapper(lambda stdscr: train_and_play(stdscr))
