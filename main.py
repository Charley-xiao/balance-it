import pygame
import sys
import math
import numpy as np
# import cupy as cp
from constants import *

pygame.init()

velocity = np.zeros((int(WIDTH / PIXEL_SIZE), int(HEIGHT / PIXEL_SIZE), 2)).astype(np.float32) #0: X; 1: Y
# velocity = cp.asarray(velocity)
# Game Objective: Balance the ball, whose velocity is affected by the wind, within the screen.
ball_position = [WIDTH / 2., HEIGHT / 2.]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Balance It! A Python game for CFD")

clock = pygame.time.Clock()

def draw_circle(screen, color, center, radius):
    pygame.draw.circle(screen, color, center, radius)

def draw_square(screen, color, center, side_length):
    pygame.draw.rect(screen, color, (center[0] - side_length // 2, center[1] - side_length // 2, side_length, side_length))

def get_color_from_velocity(x, y):
    vtmp = velocity[x, y]
    if vtmp[0] > MAX_VELOCITY:
        vtmp[0] = MAX_VELOCITY
    elif vtmp[0] < -MAX_VELOCITY:
        vtmp[0] = -MAX_VELOCITY
    if vtmp[1] > MAX_VELOCITY:
        vtmp[1] = MAX_VELOCITY
    elif vtmp[1] < -MAX_VELOCITY:
        vtmp[1] = -MAX_VELOCITY
    v_tot = math.sqrt(vtmp[0] ** 2 + vtmp[1] ** 2)
    if v_tot > MAX_VELOCITY:
        v_tot = MAX_VELOCITY
    elif v_tot < -MAX_VELOCITY:
        v_tot = -MAX_VELOCITY
    return (min(255,int(255 * abs(v_tot) / MAX_VELOCITY)), 0, min(255,int(255 * (1 - abs(v_tot) / MAX_VELOCITY))))

def render_velocity(screen):
    for x in range(0, WIDTH, PIXEL_SIZE):
        for y in range(0, HEIGHT, PIXEL_SIZE):
            draw_circle(screen, get_color_from_velocity(x // PIXEL_SIZE, y // PIXEL_SIZE), (x, y), PIXEL_SIZE)

def update_ball_position(ball_position, velocity, cursor_position):
    ball_position[0] += velocity[int(ball_position[0]) // PIXEL_SIZE, int(ball_position[1]) // PIXEL_SIZE, 0] * BASE_SPEED
    ball_position[1] += velocity[int(ball_position[0]) // PIXEL_SIZE, int(ball_position[1]) // PIXEL_SIZE, 1] * BASE_SPEED
    # If it collides with the cursor, then it bounces off from the circle with the cursor as the center.
    if math.sqrt((ball_position[0] - cursor_position[0]) ** 2 + (ball_position[1] - cursor_position[1]) ** 2) <= RADIUS_CURSOR + RADIUS_BALL:
        ball_position[0] = (cursor_position[0] + (ball_position[0] - cursor_position[0]) * (RADIUS_CURSOR + RADIUS_BALL) / math.sqrt((ball_position[0] - cursor_position[0]) ** 2 + (ball_position[1] - cursor_position[1]) ** 2))
        ball_position[1] = (cursor_position[1] + (ball_position[1] - cursor_position[1]) * (RADIUS_CURSOR + RADIUS_BALL) / math.sqrt((ball_position[0] - cursor_position[0]) ** 2 + (ball_position[1] - cursor_position[1]) ** 2))
    

def init(ball_position, velocity):
    velocity[:, :, 0] = WIND_VELOCITY_X
    velocity[:, :, 1] = WIND_VELOCITY_Y
    ball_position[0] = WIDTH // 3
    ball_position[1] = HEIGHT // 2

def gameover_screen():
    font = pygame.font.Font(None, 36)
    gameover_text = font.render("Game Over!", True, RED)
    instructions_text = font.render("Press Q to quit or R to restart", True, BLACK)
    screen.blit(gameover_text, (WIDTH // 2 - gameover_text.get_width() // 2, HEIGHT // 2 - 50))
    screen.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 2 + 20))
    pygame.display.flip()

def menu_screen():
    font = pygame.font.Font(None, 36)
    menu_text = font.render("Balance It!", True, BLACK)
    options_text = font.render("Press SPACE to play or 1 to test or 2 in pure mode", True, BLACK)
    screen.blit(menu_text, (WIDTH // 2 - menu_text.get_width() // 2, HEIGHT // 2 - 50))
    screen.blit(options_text, (WIDTH // 2 - options_text.get_width() // 2, HEIGHT // 2 + 20))
    pygame.display.flip()

####### FLOW DEFINITION #######
Re = 839.0
nx = WIDTH // PIXEL_SIZE
ny = HEIGHT // PIXEL_SIZE
q = 9
uLB = WIND_VELOCITY_X
nulb = uLB * RADIUS_CURSOR / Re
omega = 1.0 / (3.0 * nulb + 0.5)

####### LATTICE CONSTANTS #######
c = np.array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]])
t = 1. / 36. * np.ones(q)
t[np.asarray([np.linalg.norm(ci) < 1.1 for ci in c])] = 1. / 9.
t[0] = 4. / 9.
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]
i1 = np.arange(q)[np.asarray([ci[0] < 0 for ci in c])]
i2 = np.arange(q)[np.asarray([ci[0] == 0 for ci in c])]
i3 = np.arange(q)[np.asarray([ci[0] > 0 for ci in c])]

####### FUNCTION DEFINITIONS #######
sumpop = lambda fin: np.sum(fin, axis=0)
def equilibrium(rho, u):
    cu = 3.0 * np.dot(c, u.transpose(1, 0, 2))
    usqr = 3.0 / 2.0 * (u[0] ** 2 + u[1] ** 2)
    feq = np.zeros((q, nx, ny))
    for i in range(q):
        feq[i, :, :] = rho * t[i] * (1.0 + cu[i] + 0.5 * cu[i] ** 2 - usqr)
    return feq

vel = np.fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*np.sin(y/ny*2*np.pi)),(2,nx,ny))
feq = equilibrium(1.0, vel)
fin = feq.copy()

def collision_and_stream(velocity, cursor_position):
    obstacle = np.fromfunction(lambda x, y: 
                               np.logical_or(
                               np.logical_or(
                                   (x - cursor_position[0] // PIXEL_SIZE) ** 2 + (y - cursor_position[1] // PIXEL_SIZE) ** 2 <= (RADIUS_CURSOR // PIXEL_SIZE) ** 2,
                                   y >= HEIGHT // PIXEL_SIZE,
                                   y <= 0
                               ),np.logical_or(0, x >= WIDTH // PIXEL_SIZE)),
                               (nx, ny))
    fin[i1, -1, :] = fin[i1, -2, :]
    fin[i2, -1, :] = fin[i2, -2, :]
    fin[i3, -1, :] = fin[i3, -2, :]
    rho = sumpop(fin)
    u = np.dot(c.transpose(), fin.transpose((1, 0, 2))) / rho
    u[:, 0, :] = vel[:, 0, :]
    rho[0, :] = 1. / (1. - u[0, 0, :]) * (sumpop(fin[i2, 0, :]) + 2. * sumpop(fin[i1, 0, :]))
    rho[rho >= MAX_VALUE] = MAX_VALUE
    rho[rho <= 0] = 0.1
    feq = equilibrium(rho, u)
    if UNDER_TEST:
        print(rho)
    fin[i3, 0, :] = fin[i1, 0, :] + feq[i3, 0, :] - fin[i1, 0, :]
    fout = fin - omega * (fin - feq)
    for i in range(q):
        fout[i, obstacle] = fin[noslip[i], obstacle]
    for i in range(q):
        fin[i, :, :] = np.roll(np.roll(fout[i, :, :], c[i, 0], axis=0), c[i, 1], axis=1)
    velocity[:, :, 0] = u[1, :, :]
    velocity[:, :, 1] = u[0, :, :]
    # velocity[obstacle] = MAX_VELOCITY

def main():
    cursor_position = [WIDTH // 2, HEIGHT // 2]
    game_over = False
    init(ball_position, velocity)
    survive_time = 0
    at_main_menu = True
    render_ball = True
    pure_mode = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if UNDER_TEST:
                    with open('velocity.txt', 'w') as f:
                        for i in range(velocity.shape[0]):
                            for j in range(velocity.shape[1]):
                                f.write(f'{velocity[i,j,0]},{velocity[i,j,1]}\n')
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

            if at_main_menu and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    at_main_menu = False
                    render_ball = True
                    pure_mode = False
                elif event.key == pygame.K_1:
                    at_main_menu = False
                    render_ball = False
                    pure_mode = False
                elif event.key == pygame.K_2:
                    at_main_menu = False
                    render_ball = False
                    pure_mode = True
            
            if (not at_main_menu) and event.type == pygame.KEYDOWN and (not game_over):
                if event.key == pygame.K_r:
                    at_main_menu = True

        screen.fill(WHITE)

        if at_main_menu:
            menu_screen()
        elif not game_over:
            for _ in range(10):
                collision_and_stream(velocity, cursor_position)
            render_velocity(screen)
            try:
                if render_ball:
                    update_ball_position(ball_position, velocity, cursor_position)
            except IndexError as e:
                print(e)
                game_over = True
            if render_ball:
                draw_circle(screen, GREEN, (int(ball_position[0]), int(ball_position[1])), RADIUS_BALL)
                if ball_position[0] < 0 or ball_position[0] > WIDTH or ball_position[1] < 0 or ball_position[1] > HEIGHT:
                    game_over = True

            draw_circle(screen, GOLD, cursor_position, RADIUS_CURSOR)
            survive_time += 1
            if not pure_mode:
                font = pygame.font.Font(None, 36)
                survive_time_text = font.render("Score: " + str(survive_time // 10), True, BLACK)
                screen.blit(survive_time_text, (10, 10))
                font2 = pygame.font.Font(None, 24)
                notice_text = font2.render("Press R to return to the main menu", True, BLACK)
                screen.blit(notice_text, (10, 30))
            if UNDER_TEST and survive_time % 10 == 0:
                print(f'Ball position: {ball_position[0]}, {ball_position[1]}')
                print(f'Sample Velocity: {velocity[30,30,0]}, {velocity[30,30,1]}')
            pygame.display.flip()
        else:
            gameover_screen()

        pygame.display.flip()

        clock.tick(FPS)

if __name__ == "__main__":
    main()