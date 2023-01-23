import pygame
import time


# define a main function
def main():
    # initialize the pygame module
    pygame.init()

    # load and set the logo
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("minimal program")

    # create a surface on screen that has the size of 240 x 180
    screen_width = 240*2
    screen_height = 180*2

    screen = pygame.display.set_mode((screen_width, screen_height))
    bgd_image = pygame.image.load("background.png")
    screen.blit(bgd_image, (0, 0))
    pygame.display.flip()

    # define the position of the smiley
    xpos = 50
    ypos = 50
    # how many pixels we move our smiley each frame
    step_x = 10
    step_y = 10

    image = pygame.image.load("01_image.png")
    image.set_colorkey((255, 0, 255))
    image.set_alpha(128)

    # define a variable to control the main loop
    running = True

    # main loop
    while running:

        if xpos > screen_width - 64 or xpos < 0:
            step_x = -step_x
        if ypos > screen_height - 64 or ypos < 0:
            step_y = -step_y

        # update the position of the smiley
        xpos += step_x  # move it to the right
        ypos += step_y  # move it down

        screen.blit(bgd_image, (0, 0))
        screen.blit(image, (xpos, ypos))
        pygame.display.flip()
        time.sleep(.1)

        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            print(event)
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
    # call the main function
    main()