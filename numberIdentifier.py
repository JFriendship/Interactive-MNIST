import pygame
import numpy as np
import torch
from ConvNetMNIST import ConvNetMNIST

# CONSTANTS
CELL_SIZE = 10
LINE_WIDTH = 2
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
GRID_SIZE = 28           # 28x28 for the model to work

# Load MNIST CNN Model
model = ConvNetMNIST()
model.load_state_dict(torch.load('./MNIST_Model.pt'))
model.eval()

# ==============================generateContainer(x,y)============================== #
# Draws a square that houses a 28 by 28 grid onto the screen
# Each cell has a pygame rectangle associated with it (even though they are squares)
# Each square's reference is stored in an array called rectangle_ref
def generateContainer(x,y):
    # Draw Border
    full_length = GRID_SIZE*CELL_SIZE
    pygame.draw.line(screen, "black", (x, y-LINE_WIDTH), (x+full_length, y-LINE_WIDTH), width=LINE_WIDTH)   # TOP
    pygame.draw.line(screen, "black", (x-LINE_WIDTH, y), (x-LINE_WIDTH, y+full_length), width=LINE_WIDTH)   # LEFT
    pygame.draw.line(screen, "black", (x, y+full_length), (x+full_length, y+full_length), width=LINE_WIDTH)   # BOTTOM
    pygame.draw.line(screen, "black", (x+full_length, y), (x+full_length, y+full_length), width=LINE_WIDTH)   # RIGHT

    # Create squares for the container
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rectangle_ref.append(pygame.Rect(x+(j*CELL_SIZE), y+(i*CELL_SIZE), CELL_SIZE, CELL_SIZE))

# Resets the container values to 0 via the pixel_pos array
# Changes the colour of the cells back to the background colour
def resetBoard():
    for i, cell in enumerate(rectangle_ref):
        pixel_pos[int(i/GRID_SIZE), i%GRID_SIZE] = 0
        pygame.draw.rect(screen, (73, 92, 91), cell)

def resetPredictionText():
    predictionText = 'Prediction: '
    predictionTextRender = font.render(predictionText, True, (255,255,255))
    screen.blit(predictionTextRender, predictionTextRef)

rectangle_ref = [] 
pixel_pos = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
running = True
deltaTime = 0
timer = 0

pygame.init()

predictionText = 'Prediction: '
font = pygame.font.SysFont('arial', 24)
clearText = font.render('Clear', True, (0,0,0))
evaluateText = font.render('Evaluate', True, (255,255,255))
predictionTextRender = font.render(predictionText, True, (255,255,255))


pygame.display.set_caption('Number Identifier')
screen.fill((73, 92, 91))   # Set background colour

x_start = SCREEN_WIDTH/2 - (GRID_SIZE/2) * CELL_SIZE
y_start = SCREEN_HEIGHT/2 - (GRID_SIZE/2) * CELL_SIZE

# Generate Board
generateContainer(x_start, y_start)
resetButtonRef = pygame.Rect(x_start, y_start+(GRID_SIZE*CELL_SIZE) + CELL_SIZE*2, GRID_SIZE*CELL_SIZE, CELL_SIZE*3)
pygame.draw.rect(screen, (232, 217, 86), resetButtonRef)
evaluateButtonRef = pygame.Rect(x_start, y_start+(GRID_SIZE*CELL_SIZE) + CELL_SIZE*5, GRID_SIZE*CELL_SIZE, CELL_SIZE*3)
pygame.draw.rect(screen, (0, 0, 0), evaluateButtonRef)
predictionTextRef = pygame.Rect(x_start, y_start-CELL_SIZE*3, GRID_SIZE*CELL_SIZE, CELL_SIZE*3-5)


# Set mouse button toggle
mouseButtonFlag = False

while running:
    screen.blit(clearText, resetButtonRef)
    screen.blit(evaluateText, evaluateButtonRef)
    screen.blit(predictionTextRender, predictionTextRef)

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            mouseButtonFlag = False

        elif event.type == pygame.MOUSEBUTTONDOWN or mouseButtonFlag:
            mouseButtonFlag = True
            x, y = event.pos
            # Check for reset button press
            if event.type == pygame.MOUSEBUTTONDOWN and resetButtonRef.collidepoint(x,y):
                resetBoard()
            
            # Check for evaluate button press
            if event.type == pygame.MOUSEBUTTONDOWN and evaluateButtonRef.collidepoint(x,y):
                test_pixel_pos = pixel_pos.reshape(1,1,28,28)
                pygame.draw.rect(screen, (73, 92, 91), predictionTextRef)
                predictionText = 'Prediction: ' + str(torch.argmax(model(torch.from_numpy(test_pixel_pos)), 1).item())
                predictionTextRender = font.render(predictionText, True, (255,255,255))
                screen.blit(predictionTextRender, predictionTextRef)

            for i, cell in enumerate(rectangle_ref):
                if cell.collidepoint(x,y):
                    pygame.draw.rect(screen, (0, 0, 0), cell)
                    pixel_pos[int(i/GRID_SIZE), i%GRID_SIZE] = 1

        elif event.type == pygame.QUIT:
            running = False

        
# flip() the display to put your work on screen
    pygame.display.flip()

    deltaTime = clock.tick(60) / 1000 # convert to seconds

pygame.quit()