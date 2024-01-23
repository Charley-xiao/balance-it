import pygame
import sys
import math
import numpy as np
from constants import *
from lbm import collision_and_stream

pygame.init()

velocity = np.full((int(WIDTH / PIXEL_SIZE), int(HEIGHT / PIXEL_SIZE), 2), WIND_VELOCITY)
# Game Objective: Balance the ball, whose velocity is affected by the wind, within the screen.
ball_position = [WIDTH // 2, HEIGHT // 2]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Balance It! A Python game for CFD")

clock = pygame.time.Clock()

def draw_circle(screen, color, center, radius):
    pygame.draw.circle(screen, color, center, radius)

def draw_square(screen, color, center, side_length):
    pygame.draw.rect(screen, color, (center[0] - side_length // 2, center[1] - side_length // 2, side_length, side_length))

def get_color_from_velocity(x, y):
    return (255 * velocity[x, y, 0] // MAX_VELOCITY, 0, 255 * velocity[x, y, 1] // MAX_VELOCITY)

def render_velocity(screen):
    for x in range(0, WIDTH, PIXEL_SIZE):
        for y in range(0, HEIGHT, PIXEL_SIZE):
            draw_circle(screen, get_color_from_velocity(x // PIXEL_SIZE, y // PIXEL_SIZE), (x, y), PIXEL_SIZE)

def update_ball_position(ball_position, velocity, cursor_position):
    ball_position[0] += int(velocity[ball_position[0] // PIXEL_SIZE, ball_position[1] // PIXEL_SIZE, 0])
    ball_position[1] += int(velocity[ball_position[0] // PIXEL_SIZE, ball_position[1] // PIXEL_SIZE, 1])
    # If it collides with the cursor, then it bounces off.
    if abs(ball_position[0] - cursor_position[0]) <= RADIUS + SIDE_LENGTH // 2 and abs(ball_position[1] - cursor_position[1]) <= RADIUS + SIDE_LENGTH // 2:
        ball_position[0] -= 2 * int(velocity[ball_position[0] // PIXEL_SIZE, ball_position[1] // PIXEL_SIZE, 0])
    if abs(ball_position[1] - cursor_position[1]) <= RADIUS + SIDE_LENGTH // 2 and abs(ball_position[0] - cursor_position[0]) <= RADIUS + SIDE_LENGTH // 2:
        ball_position[1] -= 2 * int(velocity[ball_position[0] // PIXEL_SIZE, ball_position[1] // PIXEL_SIZE, 1])

def init(ball_position, velocity):
    velocity[:, :, 0] = WIND_VELOCITY
    velocity[:, :, 1] = 0
    ball_position[0] = WIDTH // 3
    ball_position[1] = HEIGHT // 2

def gameover_screen():
    font = pygame.font.Font(None, 36)
    gameover_text = font.render("Game Over!", True, RED)
    instructions_text = font.render("Press Q to quit or R to restart", True, BLACK)
    screen.blit(gameover_text, (WIDTH // 2 - gameover_text.get_width() // 2, HEIGHT // 2 - 50))
    screen.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 2 + 20))
    pygame.display.flip()

def main():
    cursor_position = [WIDTH // 2, HEIGHT // 2]
    game_over = False
    init(ball_position, velocity)
    survive_time = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                cursor_position = event.pos

            if game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_r:
                    game_over = False
                    init(ball_position, velocity)

        screen.fill(WHITE)
        
        if not game_over:
            collision_and_stream(velocity, cursor_position)
            render_velocity(screen)
            try:
                update_ball_position(ball_position, velocity, cursor_position)
            except IndexError:
                game_over = True
            draw_circle(screen, GREEN, ball_position, RADIUS)

            if ball_position[0] < 0 or ball_position[0] > WIDTH or ball_position[1] < 0 or ball_position[1] > HEIGHT:
                game_over = True

            draw_square(screen, BLUE, cursor_position, SIDE_LENGTH)
            survive_time += 1
            font = pygame.font.Font(None, 36)
            survive_time_text = font.render("Score: " + str(survive_time // 10), True, BLACK)
            screen.blit(survive_time_text, (10, 10))
            pygame.display.flip()
        else:
            gameover_screen()

        pygame.display.flip()

        clock.tick(FPS)

if __name__ == "__main__":
    main()