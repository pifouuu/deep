from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

import random
import numpy as np
import math



HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.00025

# CAREFUL : CHECK THAT IT IS THE PROPER HUBER LOSS
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

def processImage(img):

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance

    o = gray.astype('float32') / 128 - 1    # normalize
    return o

class Brain:
    def __init__(self, stateCnt, actionCnt, learning_rate):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.learning_rate = learning_rate

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.stateCnt), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))

        model.add(Dense(units=self.actionCnt, activation='linear'))

        opt = Adam(lr=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt[0], self.stateCnt[1], self.stateCnt[2]), target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 50000

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1

EXPLORATION_STOP = 10000  # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 100


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt, learning_rate):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.learning_rate = learning_rate

        self.brain = Brain(self.stateCnt, self.actionCnt,self.learning_rate)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = np.zeros(self.stateCnt)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((len(batch), self.stateCnt[0], self.stateCnt[1], self.stateCnt[2]))
        y = np.zeros((len(batch), self.actionCnt))

        for i in range(len(batch)):
            o = batch[i]
            s = o[0];
            a = o[1];
            r = o[2];
            s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(pTarget_[i])

            x[i] = s
            y[i] = t

        return (x, y)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y = self._getTargets(batch)
        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)
        self.exp += 1

    def replay(self):
        pass


# -------------------- ENVIRONMENT ---------------------

from gridworld import gameEnv
class Environment:
    def __init__(self, size, max_steps, total_steps):
        self.env = gameEnv(partial=False, size=size)
        self.max_steps = max_steps
        self.total_steps = total_steps

    def run(self, agent):
        img = self.env.reset()
        w = processImage(img)
        s = np.array([w])

        R = 0
        for step in range(self.max_steps):
            self.total_steps += 1
            # self.env.render()
            a = agent.act(s)

            img, r, done = self.env.step(a)
            s_ = np.array([processImage(img)])  # last two screens

            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            if self.total_steps % UPDATE_FREQ == 0:
                agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 1
env = Environment(5, 50, 0)
stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
actionCnt = env.env.actions
LEARNING_RATE = 0.0001
NUM_EPISODES = 100
UPDATE_FREQ = 4

agent = Agent(stateCnt, actionCnt, LEARNING_RATE)
randomAgent = RandomAgent(actionCnt)
print("Initialization with random agent...")
while randomAgent.exp < MEMORY_CAPACITY:
    env.run(randomAgent)
    print(randomAgent.exp, "/", MEMORY_CAPACITY)

agent.memory = randomAgent.memory

randomAgent = None

print("Starting learning")
for episode in range(NUM_EPISODES):
    env.run(agent)
agent.brain.model.save("gridworld")