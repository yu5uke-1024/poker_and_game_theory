# --- library ---
import numpy as np
import pygame
from pygame.locals import *
import sys

def initialize():

    # 初期化
    pygame.init()
    pygame.display.set_mode((1200, 800))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Kuhn Poker")
    screen.fill((0,150,0))
    font = pygame.font.Font(None, 30)

    text = font.render("START", True, (255,255,255))   # 描画する文字列の設定
    screen.blit(text, [550, 400])# 文字列の表示位置

    num_count = 0
    game_state = "START"


    while True:
        pygame.display.update()

        if game_state == "START":
            text = font.render("{}".format(num_count), True, (255,255,255))
            screen.blit(text, [20, 20])
            game_state = "TIME"



        for event in pygame.event.get():

            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    num_count += 1
                    game_state = "START"
                    screen.fill((0,150,0))



initialize()
