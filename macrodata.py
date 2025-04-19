####################################################################################################
# macrodata.py | Authors: dcalco                                         `                         #
#                                                                                                  #
# The program is mysterious and important.                                                         #
# Watch Severance!                                                                                 #
####################################################################################################

####################################################################################################
# Imports                                                                                          #
####################################################################################################
import pygame
import random
import sys
import math
import time
import timeit

####################################################################################################
# Constants                                                                                        #
####################################################################################################
# Window dimensions
WIDTH, HEIGHT = 1400, 1050
GRID_SIZE = 20
CELL_SIZE = WIDTH // GRID_SIZE
BIN_COUNT = 4  # Number of bins

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
OFF_WHITE = (237, 253, 255)

# Vibrato
MAX_MAGNITUDE = 3
VIBRATORS = 10
VIBRATION_DURATION = 40

# Bin dimensions
BIN_WIDTH = WIDTH // BIN_COUNT
BIN_HEIGHT = 60
OFFSET = CELL_SIZE*2-1
BIN_HEIGHT = 40
BIN_WIDTH = 300

LUMON_PNG = pygame.image.load("graphics/severance/luMon.png")
LUMON_PNG = pygame.transform.scale_by(LUMON_PNG, .12)

####################################################################################################
# Globals                                                                                          #
####################################################################################################
# Initialize Pygame
pygame.init()

# Set up the display
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Font
font = pygame.font.SysFont('Arial', 30)

# Create a clock object
clock = pygame.time.Clock()


# Generate grid with random numbers
numbers = [[random.randint(0, 9) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
vibrating_indices = random.sample(range(GRID_SIZE * GRID_SIZE), VIBRATORS)
vibrato_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]        # List to track who is a vibrator by proxy
vibration_timers = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]    # 2D list to track vibration timers for each number

selection_start = None
selection_rect = None

clock = pygame.time.Clock()
prev_time = time.time()

# Create bin rectangles
bins = [pygame.Rect(i * BIN_WIDTH, HEIGHT - BIN_HEIGHT, BIN_WIDTH, BIN_HEIGHT) for i in range(BIN_COUNT)]

def vibrato(magnitude:int, x, y):
    global vibrato_grid
    if(magnitude==0):
        return
    if(x<0 or x>=GRID_SIZE or y<0 or y>=GRID_SIZE):
        return
    vibrato_grid[x][y]=magnitude
    vibrato(magnitude-1, x-1, y)
    vibrato(magnitude-1, x+1, y)
    vibrato(magnitude-1, x, y-1)
    vibrato(magnitude-1, x, y+1)
    return

# Function to calculate distance between two points
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Utility function to draw the grid of numbers
def draw_grid(mouse_pos):
    global vibrato_grid
    vibrato_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = j * CELL_SIZE
            y = i * CELL_SIZE   
            if i * GRID_SIZE + j in vibrating_indices:
                vibrato(MAX_MAGNITUDE, i,j)
                
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
                x = j * CELL_SIZE
                y = i * CELL_SIZE  
                if (y>OFFSET and y<HEIGHT-(2*OFFSET)):
                    num = numbers[i][j]
                    label = font.render(str(num), True, OFF_WHITE)
                    if distance(mouse_pos[0], mouse_pos[1], x,y) < 100:
                        vibration_timers[i][j] = VIBRATION_DURATION
                        x += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        y += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        label = pygame.font.SysFont('Arial', 30+6*vibrato_grid[i][j]).render(str(num), True, WHITE)
                    elif vibration_timers[i][j]:
                        # Apply vibration if the timer is active
                        x += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        y += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        if (vibration_timers[i][j]>0):
                            vibration_timers[i][j] -= 1
                        label = pygame.font.SysFont('Arial', 30+6*vibrato_grid[i][j]).render(str(num), True, WHITE)
                    window.blit(label, (x + CELL_SIZE // 4, y + CELL_SIZE // 4))

def draw_scanlines():
    scanline_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    scanline_color = (182, 255, 255, 30)  # Black with low opacity for subtle scanlines
    scanline_height = 4  # Height of each scanline
    scanline_width = 4  # Height of each scanline
    for y in range(0, HEIGHT, scanline_height * 2):
        pygame.draw.rect(scanline_surface, scanline_color, pygame.Rect(0, y, WIDTH, scanline_height))
    # Create a surface for scanlines
    scanline_color = (255, 255, 255, 20)  # Black with low opacity for subtle scanlines
    
    for y in range(0, WIDTH, scanline_width * 2):
        pygame.draw.rect(scanline_surface, scanline_color, pygame.Rect(y, 0, scanline_width, HEIGHT))
    # Overlay the scanlines at the very end
    window.blit(scanline_surface, (0, 0))#, special_flags=pygame.BLEND_ADD)

def draw_bg():
    window.fill(BLACK)
    #HEADER
    pygame.draw.line(window, OFF_WHITE, (0,OFFSET) , (WIDTH, OFFSET), 3)
    pygame.draw.line(window, OFF_WHITE, (0,OFFSET-15) , (WIDTH, OFFSET-15), 3)
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(82,33,1200,60), 3, -1 )
    window.blit(LUMON_PNG, (1150,0))
    #TOP BOX
    pygame.draw.line(window, OFF_WHITE, (0,HEIGHT-1.5*OFFSET+15) , (WIDTH, HEIGHT-1.5*OFFSET+15), 3)
    pygame.draw.line(window, OFF_WHITE, (0,HEIGHT-1.5*OFFSET+2*15) , (WIDTH, HEIGHT-1.5*OFFSET+30), 3)
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(150,880,300,50), 3, -1 )
    txt = font.render('01', True, OFF_WHITE)
    window.blit(txt, (150+300/2-12, 880+50/2-20))
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(250+BIN_WIDTH,880,300,50), 3, -1 )
    txt = font.render('02', True, OFF_WHITE)
    window.blit(txt, (250+BIN_WIDTH+300/2-12, 880+50/2-20))
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(350+2*+BIN_WIDTH,880,300,50), 3, -1 )
    txt = font.render('03', True, OFF_WHITE)
    window.blit(txt, (350+BIN_WIDTH*2+300/2-12, 880+50/2-20))
    nxtboxoffset=9
    #PROGRESS BOXES
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(150,880+50+nxtboxoffset,300,40), 3, -1 )
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(250+BIN_WIDTH,880+50+nxtboxoffset,300,40), 3, -1 )
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(350+2*+BIN_WIDTH,880+50+nxtboxoffset,300,40), 3, -1 )
    
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(0,880+50+nxtboxoffset+40+10,WIDTH,100), 100, -1)
    hex = font.render('0x6AF307 : 0x38A687', True, BLACK)
    window.blit(hex, (566, 993))
    
def light_mode():
    pixels = pygame.surfarray.pixels2d(window)
    pixels ^= 2 ** 32 - 1
    del pixels

def game_loop() -> None:
    global selection_rect, selection_start, prev_time

    dt = time.time() - prev_time
    prev_time = time.time()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                selection_start = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                selection_start = None
                selection_rect = None
        elif event.type == pygame.MOUSEMOTION:
            if selection_start:
                x1, y1 = selection_start
                x2, y2 = event.pos
                selection_rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
        elif event.type == pygame.QUIT:
            quit = True
            pygame.quit()
            sys.exit()

    
    # Get the current mouse position
    mouse_pos = pygame.mouse.get_pos()
    
    draw_grid(mouse_pos)
    
    if selection_rect:
        pygame.draw.rect(window, OFF_WHITE, selection_rect, 2)

    # Render FPS
    '''fps = clock.get_fps()
    fps_text = font.render(f'FPS: {int(fps)}', True, BLACK)
    window.blit(fps_text, (10, HEIGHT - BIN_HEIGHT - 20))'''

    #draw_scanlines()
    
    ## LIGHT MODE
    #light_mode()

    pygame.display.flip()

    # Cap the frame rate to 60 frames per second
    clock.tick(30)

while True:
    draw_bg()
    game_loop()