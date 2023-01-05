import pygame


def init():
    pygame.init()
    window = pygame.display.set_mode((400, 400))


def getKey(keyName):
    answer = False
    for events in pygame.event.get():
        pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if keyInput[myKey]:
        answer = True
    pygame.display.update()
    return answer


def main():
    if getKey("LEFT"):
        print("Left key pressed")

    if getKey("RIGHT"):
        print("Right key Pressed")


if __name__ == '__main__':
    init()
    while True:
        main()

