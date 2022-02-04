import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, BLOCK_SIZE, Point, Directions
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.89 # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 128, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    
    def get_state(self, game):
        head = game.snake[0]

        # Get all possible states
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = (game.direction == Directions.LEFT)
        dir_r = (game.direction == Directions.RIGHT)
        dir_u = (game.direction == Directions.UP)
        dir_d = (game.direction == Directions.DOWN)

        states = [
            # Danger Straight
            (dir_r and game.is_game_over(point_r)) or
            (dir_l and game.is_game_over(point_l)) or
            (dir_u and game.is_game_over(point_u)) or
            (dir_d and game.is_game_over(point_d)),

            # Danger Right
            (dir_r and game.is_game_over(point_d)) or
            (dir_l and game.is_game_over(point_u)) or
            (dir_u and game.is_game_over(point_r)) or
            (dir_d and game.is_game_over(point_l)),
        
            # Danger Left
            (dir_r and game.is_game_over(point_u)) or
            (dir_l and game.is_game_over(point_d)) or
            (dir_u and game.is_game_over(point_l)) or
            (dir_d and game.is_game_over(point_r)),

            # Move Directions
            dir_l, dir_r, dir_u, dir_d,

            # Food Directions
            (game.food.x < head.x), # Food left
            (game.food.x > head.x), # Food right
            (game.food.y < head.y), # Food up
            (game.food.y > head.y), # Food down
        ]

        return np.array(states, dtype=np.int8)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves => tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if (random.randint(0,200) < self.epsilon):
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_means = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # Get the current state
        state_old = agent.get_state(game)

        # Depending on the state, get the move
        final_move = agent.get_action(state_old)

        # Apply the move and get new state
        reward, done, score = game.play(final_move)

        # Get the new state
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short(state_old, final_move, reward, state_new, done)

        # Remember the move
        agent.remember(state_old, final_move, reward, state_new, done)

        # Train long memory
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if (score > record) :
                record = score
                agent.model.save();

            print("Games: {} | Record: {}".format(agent.n_games, record))  

            plot_scores.append(score)
            total_score += score
            plot_means.append(total_score / agent.n_games)
            plot(plot_scores, plot_means)

if __name__ == "__main__":
    train()