import nintaco
import pygame
import numpy as np
import cv2
from PIL import ImageGrab
import time
from keras.models import load_model
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin'
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


GamepadButtons = {
     'A': 0,
     'B': 1,
     'Select': 2,
     'Start': 3,
     'Up': 4,
     'Down': 5,
     'Left': 6,
     'Right': 7
}
file_name = ''
alive = True
Pressed_A = False
Pressed_B = False
Pressed_Left = False
Pressed_Right = False
Pressed_Start = False
Pressed_Select = False

last_time = time.time()
clock = pygame.time.Clock()
start = False
training = []
frame_count = 0
stateStartFrame = -1
timer = 0
counter = 0
nintaco.initRemoteAPI("localhost", 9999)

api = nintaco.getAPI()
AI = load_model('Learning_final_Save.model')#Model to start
plot_model(AI, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
history_array = []

eps = 0.5
decay_factor = 0.999
iteration = 0
success = 0
loss = 0

height_count = 0
highest_height = 192
last_height = 200;
highest_level = 0;

batch_states = []
last_state = -1
last_action = -1
last_action_array = []
batch_counter = 0
run_batch = False
run_bad_batch = False




def screen_record():
    global last_time
    printscreen = np.array(ImageGrab.grab(bbox=(100,130,700,640)))
    last_time = time.time()
    processed = greycode(printscreen)
    cv2.imshow('AIBOX', processed)
    cv2.moveWindow("AIBOX", 500, 150);
    processed = cv2.resize(processed, (200, 140))

    #FOR MANUAL TRAINING SETS
    #training.append([processed, check_input()])

    processed = np.array(processed).reshape(-1, 200, 140, 1)
    ai_prediction(processed, True)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def greycode(screen):
    greymap = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) #changes it to greyscale
    greymap = cv2.Canny(greymap, threshold1=200, threshold2=300) #detects the raw edges of the game
    return  greymap

def ai_prediction(input, train):
    global eps
    global alive
    global last_state
    global last_action
    global last_action_array
    global success
    global loss
    global run_batch
    global run_bad_batch
    global batch_states

    if (np.random.random() <= eps or last_action == -1) and train == True:
        action = np.random.randint(0, 4, size=1)
        action = action[0]
        #print ('RANDOM ACTION', action)
    else:
        action = AI.predict(input)
        action_array = action[0]
        action = np.argmax(action[0])
        #print ('PREDICTED ACTION', action)
    print 'ACTION', action
    #Input Action
    AI_Control_Access(action)
    if (train == False):
        return
    if(last_action == -1):
        last_action = action
        last_state = input
        last_action_array = [0,0,0,0]
        return

    #convert to network outputs for training
    if (action == 0):
        action_array = [1, 0, 0, 0]
    if (action == 1):
        action_array = [0, 1, 0, 0]
    if (action == 2):
        action_array = [0, 0, 1, 0]
    if (action == 3):
        action_array = [0, 0, 0, 1]

    #see if we got higher
    reward = check_height()
    if reward == 1:
        success += 1
        print ('Rewarded')
        run_batch = True
    elif reward == -1:
        loss += 1
        run_bad_batch = True
        print ('Punished')

    #Don't Reward for doing nothing
    if(last_action != 0 and run_batch == False and run_bad_batch == False):
        batch_states.append([last_state, last_action_array, last_action, reward])
    #Frame Sequences
    if len(batch_states) == 9:
        batch_states.pop(0)

    #Trainng rewarded outcome
    if(run_batch and len(batch_states) == 8):
        count = 0
        for i in batch_states:
            x_train = np.array([i[0]]).reshape(-1, 200, 140, 1)
            y_train = np.asarray([i[1]])
            print count, y_train
            if 3 < count < 8:
                history = AI.fit(x_train, y_train, batch_size=1, verbose=True)
                print ('TRAINED', y_train)
                history_array.append([history.history['loss'], history.history['mean_absolute_error']])
                visual_fit_history()
            count += 1
        # x_train = np.array([i[0] for i in batch_states]).reshape(-1, 200, 140, 1)
        # y_train = np.asarray([i[1] for i in batch_states])
        # AI.fit(x_train, y_train, batch_size=1, verbose=True, epochs=8)
        batch_states = []
        AI.summary()
        run_batch = False
        AI.save('Learning_final_Save.model')
    #Unlearn bad behaviors
    elif (run_bad_batch and len(batch_states) == 8):
        count = 0
        for i in batch_states:
            x_train = np.array([i[0]]).reshape(-1, 200, 140, 1)
            if 3 < count < 8:
                history = AI.fit(x_train, np.array([[0, 0, 0, 0]]), batch_size=1, verbose=True)
                history_array.append([history.history['loss'], history.history['mean_absolute_error']])
                visual_fit_history()
            count += 1
        batch_states = []
        AI.summary()
        run_bad_batch = False
        AI.save('Learning_final.model')

    last_action_array = action_array
    last_state = input
    last_action = action

def visual_fit_history():
    np.save('history', history_array)
    hist = np.load('history.npy')
    plt.plot(hist[0])
    plt.plot(hist[1])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend(['Loss', 'Mean Absolute Error'], loc='upper left')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('loss.png', dpi=100)

def launch():
    api.addFrameListener(renderFinished)
    api.addStatusListener(statusChanged)
    api.addActivateListener(apiEnabled)
    api.addDeactivateListener(apiDisabled)
    api.addStopListener(dispose)
    #api.addControllersListener(check_input)
    api.addAccessPointListener(check_level, nintaco.PostExecute, 0x0059)
    api.addAccessPointListener(check_height, nintaco.PostExecute, 0x0066)
    api.addAccessPointListener(check_lives, nintaco.PostExecute, 0x0066)
    api.run()

def apiEnabled():
    global alive
    global frame_count
    global start
    frame_count = api.getFrameCount();
    print("API enabled")
    alive = True
    if(start == False):
        startup()

def startup():
    global start
    global AI
    create_training()
    api.loadState('C:\Users\lildo\PycharmProjects\GAMEAI3\Level1.save')
    start = True

def apiDisabled():
    print("API disabled")

def dispose():
    print("API stopped")

def statusChanged(message):
    print("Status message: %s" % message)

def check_level():
    global highest_level
    level = api.peekCPU(0x0059)
    if(level > highest_level and alive == True):
        highest_level = level;

def check_height():
    global last_height
    global height_count
    global highest_height
    height = api.peekCPU32(0x0066)
    if(height == last_height):
        height_count += 1
        if (height < highest_height and height >= 0 and alive == True and  (highest_height - height == 48)):
            highest_height = height
            height_count = 0
            print (height)
            return 1
        elif(height > highest_height and height <= 192 and alive == True and (height - highest_height == 48)):
            highest_height = height
            height_count = 0
            print height
            return -1
        else:
            return 0
    else:
        height_count = 0
        last_height = height
        return 0

def check_lives():
    global alive
    lives = api.peekCPU(0x0020)
    #254 255 death counter for dying
    if(lives == 128):
        alive = False
        death_control()
    elif(lives >= 0):
        alive = True

def renderFinished():
    global stateStartFrame
    global counter
    global timer

    frame = api.getFrameCount()
    clock.tick()
    counter += 1
    if(counter == 4):
        check_level()
        check_lives()
        screen_record()
        counter = 0
    #1 minute per run
    if(timer == 3600):
        death_control()
    timer += 1

def receive_input():
    if(Pressed_A == True):
        api.writeGamepad(0, GamepadButtons['A'], True)
    else:
        api.writeGamepad(0, GamepadButtons['A'], False)

    if (Pressed_B == True):
        api.writeGamepad(0, GamepadButtons['B'], True)
    else:
        api.writeGamepad(0, GamepadButtons['B'], False)

    if (Pressed_Right == True):
        api.writeGamepad(0, GamepadButtons['Right'], True)
    else:
        api.writeGamepad(0, GamepadButtons['Right'], False)

    if (Pressed_Left == True):
        api.writeGamepad(0, GamepadButtons['Left'], True)
    else:
        api.writeGamepad(0, GamepadButtons['Left'], False)

    check_input()


def check_input():
    Check_A = api.readGamepad(0, GamepadButtons['A'])
    Check_B = api.readGamepad(0, GamepadButtons['B'])
    Check_Right = api.readGamepad(0, GamepadButtons['Right'])
    Check_Left = api.readGamepad(0, GamepadButtons['Left'])
    #print ('BUTTONS TO REGISTER PRESSED', Check_A, Check_B, Check_Left, Check_Right)
    return (Check_A, Check_B, Check_Left, Check_Right)

def AI_Control_Access(input):
    global Pressed_A
    global Pressed_B
    global Pressed_Left
    global Pressed_Right

    if(input == 0):
        Pressed_Left = False
        Pressed_Right = False
        Pressed_A = False
    elif(input == 1):
        Pressed_Left = False
        Pressed_Right = False
        Pressed_A = True
    elif (input == 2):
        Pressed_Left = True
        Pressed_Right = False
        Pressed_A = False
    elif (input == 3):
        Pressed_Left = False
        Pressed_Right = True
        Pressed_A = False
    #print ('BUTTON TO PRESS', Pressed_A,Pressed_B,Pressed_Left,Pressed_Right)
    receive_input()


def death_control():
    global timer
    global last_height
    last_height = 200
    #print (file_name, 'SAVED TRAINING')

    timer = 0
    reset()

def reset():
    global eps
    global iteration
    global alive
    global last_action
    global batch_states
    global highest_height
    eps *= decay_factor
    iteration += 1
    highest_height = 192
    alive = True
    last_action = -1
    batch_states = []
    api.loadState('C:\Users\lildo\PycharmProjects\GAMEAI3\Level1.save')

def create_training():
    global file_name
    global training
    global history_array
    file_name = 'ICE_Train5.npy'

    if os.path.isfile(file_name):
        print('Found Training')
        training = list(np.load(file_name))
    else:
        print('Cant find')
        training = []

    if os.path.isfile('history.npy'):
        print('history')
        history_array = list(np.load(file_name))
    else:
        print('no history')
        history_array = []

if __name__ == "__main__":
    launch()

