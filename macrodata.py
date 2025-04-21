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
import cv
from typing import Any
from sgt_types import DetectedCentroids, Loop_Result_t, Centroid

from cv2.typing import MatLike

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
BLUE = (0,0,150)

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

ANIMATION_DURATION = 1000  # Total animation duration in milliseconds

VIBRATION_EXAGGERATION = 8

####################################################################################################
# Globals                                                                                          #
####################################################################################################

timer_selection = None
box_selection = False
animation_start_time = None
open_animation_active = True
selected_numbers = None


# Initialize Pygame

def init(_=None):
    global clock, window, font
    pygame.init()

    clock = pygame.time.Clock()
    cv.toggle_hover(OFFSET, 15, 15, GRID_SIZE)

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MDR")

    font = pygame.font.SysFont('Arial', 30)
    

def deinit() -> int:
    pygame.quit()
    return 42

# Font
font: pygame.font.SysFont
# Create a clock object
clock: pygame.time.Clock

        

# Generate grid with random numbers
numbers = [[random.randint(0, 9) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
vibrating_indices = random.sample(range(GRID_SIZE * GRID_SIZE), VIBRATORS)
vibrato_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]        # List to track who is a vibrator by proxy
vibration_timers = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]    # 2D list to track vibration timers for each number
animation_idx = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] 

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

'''def draw_grid():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = j * CELL_SIZE
            y = i * CELL_SIZE  
            if (y>OFFSET and y<HEIGHT-(2*OFFSET)):
                num = numbers[i][j]
                label = font.render(str(num), True, OFF_WHITE)
                if hover_grid[i][j] > 100:
                    x += random.choice([-1 * VIBRATION_EXAGGERATION, 0, VIBRATION_EXAGGERATION])
                    y += random.choice([-1 * VIBRATION_EXAGGERATION, 0, VIBRATION_EXAGGERATION])
                    label = pygame.font.SysFont('Arial', 30+VIBRATION_EXAGGERATION).render(str(num), True, WHITE)
                label_rect = label.get_rect(center=(x + CELL_SIZE / 2, y + CELL_SIZE / 2))
                window.blit(label, label_rect.topleft)'''



def top_leftmost(x1, y1, x2, y2):
    if (y1 < y2) or (y1 == y2 and x1 < x2):
        return (x1, y1)
    else:
        return (x2, y2)# Utility function to draw the grid of numbers
def oppositeTopLeft(x1, y1, x2, y2):
    if (y1 < y2) or (y1 == y2 and x1 < x2):
        return (x2, y2)
    else:
        return (x1, y1)# Utility function to draw the grid of numbers
    

def draw_grid():
    global vibrato_grid, selected_numbers
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
                    if hover_grid[i][j] > 100:
                        vibration_timers[i][j] = VIBRATION_DURATION
                        x += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        y += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        label = pygame.font.SysFont('Arial', 30+VIBRATION_EXAGGERATION*vibrato_grid[i][j]).render(str(num), True, WHITE)

                    elif vibration_timers[i][j]:
                        # Apply vibration if the timer is active
                        x += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        y += random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])
                        if (vibration_timers[i][j]>0):
                            vibration_timers[i][j] -= 1
                        label = pygame.font.SysFont('Arial', 30+VIBRATION_EXAGGERATION*vibrato_grid[i][j]).render(str(num), True, WHITE)
                    '''if good_selection and selected_numbers and (i,j) in selected_numbers:
                        label_rect = label.get_rect(center=(animation_idx[i][j][0]+20*random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]]), 
                                                            animation_idx[i][j][1]+20*random.choice([-1 * vibrato_grid[i][j], 0, vibrato_grid[i][j]])))'''
                    
                    label_rect = label.get_rect(center=(x + CELL_SIZE / 2, y + CELL_SIZE / 2))
                    window.blit(label, label_rect.topleft)
                    #window.blit(label, (x + CELL_SIZE // 4, y + CELL_SIZE // 2))


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


progressbars = [0,0,0]


def draw_bg() -> Loop_Result_t:
    res = Loop_Result_t.CONTINUE
    window.fill(BLACK)
    #HEADER
    pygame.draw.line(window, OFF_WHITE, (0,OFFSET) , (WIDTH, OFFSET), 3)
    pygame.draw.line(window, OFF_WHITE, (0,OFFSET-15) , (WIDTH, OFFSET-15), 3)
    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(82,33,1200,60), 3, -1 )
    
    avg = (progressbars[0] + progressbars[1] + progressbars[2]) /3
    thisColor = OFF_WHITE
    if ( avg >= 1):
        thisColor=BLUE
        res = Loop_Result_t.EXIT
    pygame.draw.rect(window, thisColor, pygame.Rect(82,33,avg*1200,60), 0, -1 )
    
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
    thisColor = OFF_WHITE
    if (progressbars[0] >= 1):
        thisColor=BLUE
    pygame.draw.rect(window, thisColor, pygame.Rect(150,880+50+nxtboxoffset,300,40), 3, -1 )
    pygame.draw.rect(window, thisColor, pygame.Rect(150,880+50+nxtboxoffset,min(1,progressbars[0])*300,40), 0, -1 )
    thisColor = OFF_WHITE
    if (progressbars[1] >= 1):
        thisColor=BLUE
    pygame.draw.rect(window, thisColor, pygame.Rect(250+BIN_WIDTH,880+50+nxtboxoffset,300,40), 3, -1 )
    pygame.draw.rect(window, thisColor, pygame.Rect(250+BIN_WIDTH,880+50+nxtboxoffset,min(1,progressbars[1])*300,40), 0, -1 )
    thisColor = OFF_WHITE
    if (progressbars[0] >= 1):
        thisColor=BLUE
    pygame.draw.rect(window, thisColor, pygame.Rect(350+2*+BIN_WIDTH,880+50+nxtboxoffset,300,40), 3, -1 )
    pygame.draw.rect(window, thisColor, pygame.Rect(350+2*+BIN_WIDTH,880+50+nxtboxoffset,min(1,progressbars[2])*300,40), 0, -1 )
    

    pygame.draw.rect(window, OFF_WHITE, pygame.Rect(0,880+50+nxtboxoffset+40+10,WIDTH,100), 100, -1)
    hex = font.render('0x6AF307 : 0x38A687', True, BLACK)
    window.blit(hex, (566, 993))
    return res
    
    
def light_mode():
    pixels = pygame.surfarray.pixels2d(window)
    pixels ^= 2 ** 32 - 1
    del pixels

selection_rect: pygame.Rect
selecting: bool = False

def calculate_total_magnitude(sel_idx):
    total_mag = 0
    for i,j in sel_idx:
        total_mag += 1 if vibrato_grid[i][j] else 0 
    if len(sel_idx) !=0 : 
        return total_mag/len(sel_idx) 
    else:
        return 0
    

select_silh = None
x1 = y1 = 0 
accuracy = 0
good_selection = False


def get_numbers_in_selection(selection_rect):
    global accuracy
    selected_numbers = []
    sel_idx = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = j * CELL_SIZE
            y = i * CELL_SIZE

            label = font.render(str(numbers[i][j]), True, WHITE)
            label_rect = label.get_rect(center=(x + CELL_SIZE / 2, y + CELL_SIZE / 2))
            
            # Check if the grid cell is within the selection rectangle's boundaries
            if selection_rect.contains(label_rect):
                selected_numbers.append(numbers[i][j])
                sel_idx.append( (i,j) )

                #pygame.draw.rect(window, WHITE, label_rect, 2)

    #print(calculate_total_magnitude(sel_idx))
    accuracy = calculate_total_magnitude(sel_idx)
    return sel_idx

# Function to draw text centered in a rectangle with a highlight
def draw_centered_text_with_highlight(text, rect):
    # Render the text
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=rect.center)

    # Draw the background highlight
    highlight_rect = text_rect.inflate(10, 10)  # Padding around the text
    pygame.draw.rect(window, BLACK, highlight_rect)

    # Draw the text
    window.blit(text_surface, text_rect.topleft)



def animate_box(rectDim):
    current_time = pygame.time.get_ticks()
    global animation_start_time, open_animation_active
    top_x = rectDim[0]
    top_y = rectDim[1]
    width = rectDim[2]
    height = rectDim[3]

    # Calculate the positions of the edges
    left_edge_start = (top_x, top_y)
    left_edge_end = (top_x, top_y + height)
    right_edge_start = (top_x + width, top_y)
    right_edge_end = (top_x + width, top_y + height)
    bottom_edge_start = (top_x, top_y + height)
    bottom_edge_end = (top_x + width, top_y + height)

    #pygame.draw.rect(window, OFF_WHITE, pygame.Rect(rectDim), 3, -1 )
    pygame.draw.line(window, WHITE, left_edge_start, left_edge_end, 3)
    pygame.draw.line(window, WHITE, right_edge_start, right_edge_end, 3)
    pygame.draw.line(window, WHITE, bottom_edge_start, bottom_edge_end, 3)

    if animation_start_time is not None:
        animation_elapsed = current_time - animation_start_time
        if animation_elapsed < ANIMATION_DURATION and open_animation_active:
            # Calculate how far the top edge should open
            progress = animation_elapsed / ANIMATION_DURATION
            print("animating ", progress)
            
            mid_x = (top_x+ top_x + width)/2

            # Calculate positions for the split lines
            left_line_end = (mid_x - progress * (width // 2), top_y)
            right_line_end = (mid_x + progress * (width // 2) , top_y)

            # Draw the opening box top edges
            pygame.draw.line(window, WHITE, left_edge_start, left_line_end, 3)
            pygame.draw.line(window, WHITE, right_edge_start, right_line_end, 3)
            return True
        else:
            open_animation_active = False
            return False


# Function to animate moving numbers to a target
def animate_to_capture(rectDim, mousepos):
    global animation_idx
    top_x = rectDim[0]
    top_y = rectDim[1]
    width = rectDim[2]
    height = rectDim[3]
    
    capture_x = (top_x+ top_x + width)/2
    capture_y = (top_y+ top_y + height)/2

    for (i, j) in selected_numbers:
        animation_idx[i][j] = (mousepos[0],mousepos[1])

hover_grid: MatLike
last_finger: Centroid = None
savex1 = None
savey1 = None
savex2 = None
savey2 = None
start_animation = False

def loop(centroids: DetectedCentroids, dt: float) -> Loop_Result_t:
    
    global selection_rect, selection_start, prev_time, selected_numbers
    global selecting, select_silh, x1,y1, timer_selection, accuracy, box_selection
    global animation_start_time, open_animation_active, good_selection, start_animation
    global hover_grid, last_finger, savex1,savey1,savex2,savey2, vibrating_indices, progressbars

    if draw_bg() == Loop_Result_t.EXIT:
        return Loop_Result_t.EXIT

    #centroids: DetectedCentroids = cv.cv_loop()
    
    hover_grid = cv.get_hover_grid()

    #assumed_f.xpos = x1
    #assumed_f.ypos = y1

    current_time = pygame.time.get_ticks()

    dt = time.time() - prev_time
    prev_time = time.time()

    # Get the current mouse position
    mouse_pos = pygame.mouse.get_pos()
    
    draw_grid()

    #if centroids and centroids.fingers:
    #    pygame.draw.circle(window, (0,150,150), (centroids.fingers[0].xpos,centroids.fingers[0].ypos), 10)


    if not box_selection and centroids and centroids.fingers:
        assumed_f = centroids.fingers[0]
        if last_finger is None or distance(assumed_f.xpos, assumed_f.ypos, last_finger.xpos, last_finger.ypos) > 40:
            if not selecting:
                x1 = savex1 = assumed_f.xpos
                y1 = savey1 = assumed_f.ypos
                
                #print("x1,y1:", x1,y1)
            elif selecting:    
                x2 = savex2 = assumed_f.xpos
                y2 = savey2 = assumed_f.ypos
                pygame.draw.circle(window, OFF_WHITE, (x1,y1), 10)
                actualX1, actualY1 = top_leftmost(x1,y1,x2,y2)
                actualX2, actualY2 = oppositeTopLeft(x1,y1,x2,y2)
                selection_rect = pygame.Rect(actualX1, actualY1, actualX2 - actualX1, actualY2 - actualY1)
                selected_numbers = get_numbers_in_selection(selection_rect)
                print("Selected numbers:", selected_numbers)
                timer_selection = current_time  # Start timer    
                #print("x2,y2:", x2,y2)
            selecting = not selecting
        last_finger = assumed_f
    
    if savex1 and savey1:
        pygame.draw.circle(window, BLUE, (savex1,savey1), 10)
    if savex2 and savey2:
        pygame.draw.circle(window, OFF_WHITE, (savex2,savey2), 10)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
            pygame.quit()
            sys.exit()

    if selecting and select_silh:
        pygame.draw.rect(window, OFF_WHITE, select_silh, 2)

    if not selecting and selection_rect:
        if (accuracy > 0.68):
            draw_centered_text_with_highlight("GOOD!", selection_rect)
            box_selection = True
            good_selection = True
        else:
            #draw_centered_text_with_highlight("WRONG!", selection_rect)
            if timer_selection and current_time - timer_selection > 2000:
                timer_selection = None
                selection_rect = None
                selected_numbers = None
                last_finger = None
                accuracy = 0

    if box_selection:      
        #pick box, set rectDim equal to box picked
        rectDim = (9,9,9,9)
        boxes = [ pygame.Rect(150,880,300,50), pygame.Rect(250+BIN_WIDTH,880,300,50), pygame.Rect(250+2*BIN_WIDTH,880,300,50)]
        if centroids and centroids.fingers:
            assumed_f = centroids.fingers[0]
            for idx, box in enumerate(boxes):
                if box.collidepoint(assumed_f.xpos, assumed_f.ypos):
                    rectDim= (150,880,300,50)
                    pygame.draw.rect(window, BLUE, box, 3, -1 )
                    box_selection = False
                    for i,j in selected_numbers:
                        numbers[i][j] = random.randint(0, 9)
                    vibrating_indices = random.sample(range(GRID_SIZE * GRID_SIZE), VIBRATORS)
                    timer_selection = None
                    selection_rect = None
                    selected_numbers = None
                    last_finger = None
                    accuracy = 0
                    progressbars[idx] += .25

                        

                    
                        
        
    
    #dissapear numbers once box clicked again
    #fill progress bar for box
        
    
    #regenerate new numbers and vibration indices
    
    
    if selection_rect:
        pygame.draw.rect(window, OFF_WHITE, selection_rect, 2)

    # Render FPS
    fps = clock.get_fps()
    fps_text = font.render(f'FPS: {int(fps)}', True, BLACK)
    window.blit(fps_text, (10, HEIGHT - BIN_HEIGHT - 20))

    draw_scanlines()
    
    ## LIGHT MODE
    #light_mode()

    

    pygame.display.flip()

    # Cap the frame rate to 60 frames per second
    clock.tick(30)
    return Loop_Result_t.CONTINUE


def main() -> None:
    cv.cv_init(detect_fingers=True, detect_cds=False)
    
    init()

    #loop_res: Loop_Result_t = Loop_Result_t.CONTINUE

    while True:
        draw_bg()
        loop()


    deinit()

if __name__ == "__main__":
    main()
