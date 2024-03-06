#%%
import math

import numpy as np
import pygame
import random

#%% Game 
class Snake:
    # constants
    YELLOW = (255, 255, 102)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (50, 153, 213)

    BLOCK_SIZE = 10
    DIS_WIDTH = 300
    DIS_HEIGHT = 200


    STOP_AFTER_REPETITION = 1000

    ACTION_INDEX = {
        "left": 0,
        "up": 1,
        "right": 2,
        "down": 3
    }

    def __init__(self, FRAMESPEED = 50000):
        pygame.init()
        self.dis = pygame.display.set_mode((Snake.DIS_WIDTH, Snake.DIS_HEIGHT))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.FRAMESPEED = FRAMESPEED
        self.init_new_game()

    def init_new_game(self):
        self.x1 = Snake.DIS_WIDTH / 2
        self.y1 = Snake.DIS_HEIGHT / 2
        self.snake_head = (self.x1, self.y1)
        # (pos x, pos y, action) is saved into the snake list
        self.snake_list = [(self.x1, self.y1)]
        self.action_list = [0]
        self.length_of_snake = 1

        # Create first food
        self.foodx = round(random.randrange(0, Snake.DIS_WIDTH - Snake.BLOCK_SIZE) / 10.0) * 10.0
        self.foody = round(random.randrange(0, Snake.DIS_HEIGHT - Snake.BLOCK_SIZE) / 10.0) * 10.0

        self.dead = False
        self.reason = None

        self.steps_without_food = 0

    def get_reward(self, a):
        x1_change, y1_change = self.get_changes(a)

        new_x1 = self.x1 + x1_change
        new_y1 = self.y1 + y1_change


        # Check if snake is off screen
        if new_x1 >= Snake.DIS_WIDTH or new_x1 < 0 or new_y1 >= Snake.DIS_HEIGHT or new_y1 < 0:
            # would be dead
            return -10

        # Check if snake hit tail
        if (new_x1, new_y1) in self.snake_list[1:]:
            return -10

        # Check if snake ate food
        if new_x1 == self.foodx and new_y1 == self.foody:
            return 1

        if abs(self.foodx - self.x1) > abs(self.foodx - new_x1) or abs(self.foody - self.y1) > abs(self.foody - new_y1):
            # got closer to the food
            return 0.5

        # else: small negative reward for hunger
        return -0.1

    def get_feature_representation(self):
        dist_x = self.foodx - self.snake_head[0]
        dist_y = self.foody - self.snake_head[1]

        if dist_x > 0:
            pos_x = 1  # Food is to the right of the snake
        elif dist_x < 0:
            pos_x = -1  # Food is to the left of the snake
        else:
            pos_x = 0  # Food and snake are on the same X file

        if dist_y > 0:
            pos_y = 1  # Food is below snake
        elif dist_y < 0:
            pos_y = -1  # Food is above snake
        else:
            pos_y = 0  # Food and snake are on the same Y file

        sqs = [
            (self.snake_head[0] - Snake.BLOCK_SIZE, self.snake_head[1]),
            (self.snake_head[0] + Snake.BLOCK_SIZE, self.snake_head[1]),
            (self.snake_head[0], self.snake_head[1] - Snake.BLOCK_SIZE),
            (self.snake_head[0], self.snake_head[1] + Snake.BLOCK_SIZE),
        ]

        surrounding_list = [0, 0, 0, 0]
        for i, sq in enumerate(sqs):
            if sq[0] < 0 or sq[1] < 0:  # off screen left or top
                surrounding_list[i] = 1
            elif sq[0] >= Snake.DIS_WIDTH or sq[1] >= Snake.DIS_HEIGHT:  # off screen right or bottom
                surrounding_list[i] = 1
            elif sq in self.snake_list[:-1]:  # part of tail
                surrounding_list[i] = 1
            else:
                surrounding_list[i] = 0

        game_state = (pos_x, pos_y) + tuple(surrounding_list)
        return np.array(game_state)

    def get_changes(self, action):
        if action == "left":
            x1_change = -Snake.BLOCK_SIZE
            y1_change = 0
        elif action == "right":
            x1_change = Snake.BLOCK_SIZE
            y1_change = 0
        elif action == "up":
            y1_change = -Snake.BLOCK_SIZE
            x1_change = 0
        elif action == "down":
            y1_change = Snake.BLOCK_SIZE
            x1_change = 0
        else:
            raise ValueError("invalid Action")
        return x1_change, y1_change

    def step(self, action, init_new_game_after_terminal=True):
        self.steps_without_food += 1
        # action movement
        x1_change, y1_change = self.get_changes(action)

        # Move snake
        self.x1 += x1_change
        self.y1 += y1_change
        self.snake_head = (self.x1, self.y1)
        self.snake_list.append(self.snake_head)
        self.action_list.append(Snake.ACTION_INDEX[action])

        # Check if snake is off screen
        if self.x1 >= Snake.DIS_WIDTH or self.x1 < 0 or self.y1 >= Snake.DIS_HEIGHT or self.y1 < 0:
            self.reason = 'Screen'
            self.dead = True

        # Check if snake hit tail
        if self.snake_head in self.snake_list[:-1]:
            self.reason = 'Tail'
            self.dead = True

        if self.steps_without_food >= Snake.STOP_AFTER_REPETITION:
            self.reason = "Hunger"
            self.dead = True

        # Check if snake ate food
        if self.x1 == self.foodx and self.y1 == self.foody:
            while (self.foodx, self.foody) in self.snake_list:
                self.foodx = round(random.randrange(0, Snake.DIS_WIDTH - Snake.BLOCK_SIZE) / 10.0) * 10.0
                self.foody = round(random.randrange(0, Snake.DIS_HEIGHT - Snake.BLOCK_SIZE) / 10.0) * 10.0
            self.length_of_snake += 1
            self.steps_without_food = 0

        # Delete the last cell since we just added a head for moving, unless we ate a food
        if len(self.snake_list) > self.length_of_snake:
            del self.snake_list[0]
            del self.action_list[0]

        # new state if last one was terminal
        is_terminal = False
        if self.dead:
            is_terminal = True
            self.last_score = self.length_of_snake - 1
            if init_new_game_after_terminal:
                self.init_new_game()

        # Draw food, snake and update score
        self.dis.fill(Snake.BLUE)
        self.draw_food()
        self.draw_snake()
        self.draw_score()
        pygame.display.update()

        # Next Frame
        self.clock.tick(self.FRAMESPEED)
        return is_terminal

    def draw_food(self):
        pygame.draw.rect(self.dis, Snake.GREEN, [self.foodx, self.foody, Snake.BLOCK_SIZE, Snake.BLOCK_SIZE])

    def draw_snake(self):
        for elem in self.snake_list:
            pygame.draw.rect(self.dis, Snake.BLACK, [elem[0], elem[1], Snake.BLOCK_SIZE, Snake.BLOCK_SIZE])

    def draw_score(self):
        font = pygame.font.SysFont("comicsansms", 35)
        value = font.render(f"Score: {self.length_of_snake - 1}", True, Snake.YELLOW)
        self.dis.blit(value, [0, 0])





