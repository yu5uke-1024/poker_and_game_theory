# --- library ---
import numpy as np
import pygame
from pygame.locals import *
import sys


def initialize():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Kuhn Poker")

    while True:
      pygame.display.update()

      for event in pygame.event.get():
          if event.type == QUIT:
              pygame.quit()
              sys.exit()

initialize()
