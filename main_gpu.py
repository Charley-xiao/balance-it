import pygame
import sys
import math
import numpy as np
import cupy as cp  ### CUPY CHANGE
from constants import *
# from numba import jit   # no longer strictly needed if using CuPy

pygame.init()

# --- Original Nx, Ny, and velocity definitions ---
nx = WIDTH // PIXEL_SIZE
ny = HEIGHT // PIXEL_SIZE

# Instead of a numpy array, create a CuPy array for velocity
velocity = cp.zeros((nx, ny, 2), dtype=cp.float32)  ### CUPY CHANGE

ball_position = [WIDTH / 2., HEIGHT / 2.]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Balance It! A Python game for CFD")

clock = pygame.time.Clock()

def draw_circle(screen, color, center, radius):
    pygame.draw.circle(screen, color, center, radius)

def draw_square(screen, color, center, side_length):
    pygame.draw.rect(
        screen, 
        color, 
        (center[0] - side_length // 2, center[1] - side_length // 2, side_length, side_length)
    )

def get_color_from_velocity(x, y, velocity_cpu):
    """
    We need a CPU version of velocity for color mapping (Pygame loops).
    velocity_cpu is a NumPy array (already transferred from GPU).
    """
    vtmp = velocity_cpu[x, y]
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

    # Simple color mapping from velocity magnitude to red...green.
    return (
        min(255, int(255 * abs(v_tot) / MAX_VELOCITY)),
        0,
        min(255, int(255 * (1 - abs(v_tot) / MAX_VELOCITY)))
    )

def render_velocity(screen, velocity):
    """
    For rendering, we pull velocity back to CPU memory once 
    instead of doing .get() per cell.
    """
    velocity_cpu = velocity.get()  ### CUPY CHANGE
    for x in range(0, WIDTH, PIXEL_SIZE):
        for y in range(0, HEIGHT, PIXEL_SIZE):
            color = get_color_from_velocity(x // PIXEL_SIZE, y // PIXEL_SIZE, velocity_cpu)
            draw_circle(screen, color, (x, y), PIXEL_SIZE)

def update_ball_position(ball_position, velocity, cursor_position):
    """
    velocity is on GPU, but indexing in Python with floats is tricky. 
    We'll read back 2D velocity at (ball_x, ball_y).
    """
    vx = velocity[int(ball_position[0]) // PIXEL_SIZE, int(ball_position[1]) // PIXEL_SIZE, 0].get()  ### CUPY CHANGE
    vy = velocity[int(ball_position[0]) // PIXEL_SIZE, int(ball_position[1]) // PIXEL_SIZE, 1].get()  ### CUPY CHANGE
    ball_position[0] += vx * BASE_SPEED
    ball_position[1] += vy * BASE_SPEED
    
    # If it collides with the cursor, then it bounces off from the circle
    dist = math.sqrt((ball_position[0] - cursor_position[0]) ** 2 
                     + (ball_position[1] - cursor_position[1]) ** 2)
    if dist <= RADIUS_CURSOR + RADIUS_BALL:
        # Reposition the ball just outside the cursor
        ball_position[0] = (cursor_position[0] 
                            + (ball_position[0] - cursor_position[0]) 
                            * (RADIUS_CURSOR + RADIUS_BALL) / dist)
        ball_position[1] = (cursor_position[1] 
                            + (ball_position[1] - cursor_position[1]) 
                            * (RADIUS_CURSOR + RADIUS_BALL) / dist)

def init(ball_position, velocity):
    """
    Initialize the velocity (on GPU).
    """
    velocity[:, :, 0] = WIND_VELOCITY_X
    velocity[:, :, 1] = WIND_VELOCITY_Y
    ball_position[0] = WIDTH // 3
    ball_position[1] = HEIGHT // 2

def gameover_screen():
    font = pygame.font.Font(None, 36)
    gameover_text = font.render("Game Over!", True, RED)
    instructions_text = font.render("Press Q to quit or R to restart", True, BLACK)
    screen.blit(gameover_text, 
                (WIDTH // 2 - gameover_text.get_width() // 2, 
                 HEIGHT // 2 - 50))
    screen.blit(instructions_text, 
                (WIDTH // 2 - instructions_text.get_width() // 2, 
                 HEIGHT // 2 + 20))
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
q = 9
uLB = WIND_VELOCITY_X
nulb = uLB * RADIUS_CURSOR / Re
omega = 1.0 / (3.0 * nulb + 0.5)

####### LATTICE CONSTANTS #######
# Move these to GPU as well
c_cpu = np.array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]], dtype=np.float32)
c = cp.asarray(c_cpu)  ### CUPY CHANGE
t_cpu = np.ones(q, dtype=np.float32) / 36.
t_cpu[np.asarray([np.linalg.norm(ci) < 1.1 for ci in c_cpu])] = 1./9.
t_cpu[0] = 4./9.
t = cp.asarray(t_cpu)  ### CUPY CHANGE

c_cpu_tuples = [tuple(x) for x in c_cpu]

noslip_cpu = [
    c_cpu_tuples.index(tuple(-c_cpu[i]))
    for i in range(q)
]
noslip = cp.asarray(noslip_cpu, dtype=cp.int32)  # move to GPU if needed

i1_cpu = np.arange(q)[np.asarray([ci[0] < 0 for ci in c_cpu])]
i2_cpu = np.arange(q)[np.asarray([ci[0] == 0 for ci in c_cpu])]
i3_cpu = np.arange(q)[np.asarray([ci[0] > 0 for ci in c_cpu])]
i1 = cp.asarray(i1_cpu, dtype=cp.int32)
i2 = cp.asarray(i2_cpu, dtype=cp.int32)
i3 = cp.asarray(i3_cpu, dtype=cp.int32)

# Instead of np.fromfunction for velocity initialization:
xv, yv = cp.meshgrid(cp.arange(ny), cp.arange(nx))  # watch ordering
# shape: (nx, ny) but note x=rows, y=cols in typical usage
# We'll define velocity as needed:

### CUPY CHANGE: define vel on GPU
vel = cp.zeros((2, nx, ny), dtype=cp.float32)
# replicate your formula:
vel[0, :, :] = (1 - 0) * uLB * (1.0 + 1e-4*cp.sin(yv/ny*2*cp.pi)) 
# or whichever axis you intended
# The original code: vel = np.fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*np.sin(y/ny*2*np.pi)), (2,nx,ny))
# means:
#    if d=0 => velocity_x = (1-0)*uLB*(1 + 1e-4*sin(...))
#    if d=1 => velocity_y = (1-1)*uLB*(1 + 1e-4*sin(...)) = 0
# so we replicate that logic manually as above.

# We'll define the distribution functions on GPU
def equilibrium(rho, u):
    """
    Now with CuPy arrays.
    """
    # cu shape: (q, nx, ny). But to replicate your dot logic, do:
    cu = 3.0 * cp.dot(c, u.transpose(1, 0, 2))  # shape: (q, nx, ny)
    usqr = 3.0/2.0 * (u[0]**2 + u[1]**2)
    feq_out = cp.zeros((q, nx, ny), dtype=cp.float32)
    for i in range(q):
        feq_out[i, :, :] = rho * t[i] * (1.0 + cu[i] + 0.5*cu[i]**2 - usqr)
    return feq_out

# Initial condition for fin:
feq = equilibrium(1.0, vel)
fin = feq.copy()

def collision_and_stream(velocity, cursor_position):
    """
    This is the main LBM collision+stream step, adapted to CuPy.
    """
    # Build obstacle array on GPU:
    # We can use cp.indices or meshgrid:
    # (nx, ny) => x in [0..nx-1], y in [0..ny-1]
    xx, yy = cp.meshgrid(cp.arange(ny), cp.arange(nx))
    xx = xx.astype(cp.float32)
    yy = yy.astype(cp.float32)
    
    cx = cursor_position[0] // PIXEL_SIZE
    cy = cursor_position[1] // PIXEL_SIZE
    r_cursor = RADIUS_CURSOR // PIXEL_SIZE
    
    # We'll define an obstacle if inside circle or out of domain
    # Note: watch your logic, the original code combined conditions with logical_or
    inside_circle = ( (yy - cx)**2 + (xx - cy)**2 ) <= (r_cursor**2)
    out_of_bounds = (yy >= 10*ny) | (yy < 0) | (xx >= nx) | (xx < 0)
    obstacle = inside_circle | out_of_bounds

    global fin, feq

    # Velocity inlet: replicate your logic
    fin_i1_last = fin[i1, -1, :]
    fin_i1_second_last = fin[i1, -2, :]
    fin[i2, -1, :] = fin[i2, -2, :]
    fin[i3, -1, :] = fin[i3, -2, :]
    fin[i1, -1, :] = fin_i1_second_last

    # Compute density
    rho = cp.sum(fin, axis=0)
    
    # Compute velocity: shape = (2, nx, ny)
    # u = dot(c.T, fin.T) / rho => shape of c is (9,2). 
    # But we can do a small trick with c reshapes:
    # c has shape (9,2), fin has shape (9,nx,ny).
    # We'll do: u = (2, nx, ny)
    c_t = c.transpose()  # shape (2,9)
    fin_t = fin.transpose((1,0,2))  # shape (nx,9,ny)
    # Dot: for each (nx, ny), sum over 9
    # easiest is: u[k, x, y] = sum_i( c[i, k] * fin[i, x, y] )
    # We'll do a manual approach:
    u = cp.zeros((2, nx, ny), dtype=cp.float32)
    for i in range(q):
        u[0] += c[i, 0] * fin[i]
        u[1] += c[i, 1] * fin[i]
    u /= rho

    # Force velocity at inlet
    u[:, 0, :] = vel[:, 0, :]
    
    # Fix density at inlet
    # replicate your logic:
    #   rho[0, :] = 1. / (1. - u[0, 0, :]) * ( s(i2, 0, :) + 2*s(i1, 0, :) )
    sum_i2 = cp.sum(fin[i2, 0, :], axis=0)
    sum_i1 = cp.sum(fin[i1, 0, :], axis=0)
    denom = (1. - u[0, 0, :])
    denom = cp.where(denom == 0, 1e-6, denom)  # avoid division by 0
    rho[0, :] = 1. / denom * (sum_i2 + 2.*sum_i1)
    
    # Clamping
    rho = cp.clip(rho, 0.1, MAX_VALUE)

    feq = equilibrium(rho, u)
    
    # Outflow condition
    fin[i3, 0, :] = fin[i1, 0, :] + feq[i3, 0, :] - fin[i1, 0, :]

    # Collision
    fout = fin - omega * (fin - feq)

    # Bounce-back on obstacle
    for i in range(q):
        fout[i, obstacle] = fin[noslip[i], obstacle]

    # Streaming
    for i in range(q):
        # shift by c[i,0] in x dimension, c[i,1] in y dimension
        fout_i = cp.roll(cp.roll(fout[i, :, :], int(c[i, 0]), axis=0),
                         int(c[i, 1]), axis=1)
        fin[i, :, :] = fout_i

    # Update the global velocity field in GPU array
    velocity[:, :, 0] = u[1]
    velocity[:, :, 1] = u[0]

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
                    # Write out GPU data to disk
                    velocity_cpu = velocity.get()
                    with open('velocity.txt', 'w') as f:
                        for i in range(velocity_cpu.shape[0]):
                            for j in range(velocity_cpu.shape[1]):
                                f.write(f'{velocity_cpu[i,j,0]},{velocity_cpu[i,j,1]}\n')
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

        if at_main_menu:
            screen.fill(WHITE)
            menu_screen()
        elif not game_over:
            # Do multiple sub-steps if needed
            for _ in range(10):
                collision_and_stream(velocity, cursor_position)
            
            # Render velocity field (pull back to CPU)
            screen.fill(WHITE)
            render_velocity(screen, velocity)

            if render_ball:
                try:
                    update_ball_position(ball_position, velocity, cursor_position)
                except IndexError as e:
                    print(e)
                    game_over = True
                draw_circle(screen, GREEN, (int(ball_position[0]), int(ball_position[1])), RADIUS_BALL)
                if (ball_position[0] < 0 or ball_position[0] > WIDTH or 
                    ball_position[1] < 0 or ball_position[1] > HEIGHT):
                    game_over = True

            # Draw cursor
            draw_circle(screen, GOLD, cursor_position, RADIUS_CURSOR)

            survive_time += 1
            if not pure_mode:
                font = pygame.font.Font(None, 36)
                survive_time_text = font.render(f"Score: {survive_time // 10}", True, BLACK)
                screen.blit(survive_time_text, (10, 10))
                font2 = pygame.font.Font(None, 24)
                notice_text = font2.render("Press R to return to the main menu", True, BLACK)
                screen.blit(notice_text, (10, 30))

            if UNDER_TEST and survive_time % 10 == 0:
                print(f'Ball position: {ball_position[0]}, {ball_position[1]}')
                # Example sample from the GPU velocity
                v_sample = velocity[30, 30].get()
                print(f'Sample Velocity: {v_sample[0]}, {v_sample[1]}')

            pygame.display.flip()
        else:
            screen.fill(WHITE)
            gameover_screen()

        clock.tick(FPS)

if __name__ == "__main__":
    main()
