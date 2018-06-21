import time

import gym
import numpy as np

from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GaussianNoise, Input

EPOCHS = 10
GAMES = 10
ITERATIONS = 200


def build_model():

    inp_layer = Input(shape=(96, 96, 3))
    pool1 = MaxPooling2D()(inp_layer)
    conv1 = Conv2D(16, 3, activation='relu')(pool1)
    pool2 = MaxPooling2D()(conv1)
    conv2 = Conv2D(16, 3, activation='relu')(pool2)

    flat = Flatten()(conv2)
    flat = GaussianNoise(0.1)(flat)
    flat = Dense(64)(flat)

    steer_out = Dense(10, activation='softmax')(flat)
    gas_out = Dense(10, activation='softmax')(flat)
    brake_out = Dense(10, activation='softmax')(flat)

    model = Model(inputs=inp_layer, outputs=[steer_out, gas_out, brake_out])

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model


env = gym.make('CarRacing-v0')

model = build_model()

for epoch_no in range(EPOCHS):

    t = time.time()

    inputs = []
    targets_steer = []
    targets_gas = []
    targets_brake = []

    rewards = []

    for game_no in range(GAMES):

        env.reset()
        # env.render(mode='human')

        for iter_no in range(ITERATIONS):

            if iter_no == 0:
                actions = None
            else:
                act_steer, act_gas, act_brake = model.predict(np.array([observation]))

                actions = [
                    ((act_steer.argmax() - 5) / 5.),
                    act_gas.argmax() / 10.,
                    act_brake.argmax() / 10.,
                ]

            new_observation, reward, done, info = env.step(actions)

            if iter_no != 0:
                rewards.append(reward)

                # build train

                inputs.append(new_observation)

                if reward > 0:
                    target_steer = np.zeros(10)
                    target_gas = np.zeros(10)
                    target_brake = np.zeros(10)

                    target_steer[act_steer.argmax()] = 1
                    target_gas[act_gas.argmax()] = 1
                    target_brake[act_brake.argmax()] = 1
                else:
                    target_steer = np.ones(10)
                    target_gas = np.ones(10)
                    target_brake = np.ones(10)

                    target_steer[act_steer.argmax()] = 0
                    target_gas[act_gas.argmax()] = 0
                    target_brake[act_brake.argmax()] = 1

                targets_steer.append(target_steer)
                targets_gas.append(target_gas)
                targets_brake.append(target_brake)

            observation = new_observation

    model.train_on_batch(np.array(inputs), [
        np.array(targets_steer),
        np.array(targets_gas),
        np.array(targets_brake),
    ])

    rewards = np.array(rewards)

    print('reward:', rewards.mean(), 'min/max', rewards.min(), rewards.max())

    # print(i, observation.shape, reward, done, info)
    # env.step(env.action_space.sample())
    print(time.time() - t)


# t = time.time()
# env.reset()
# for i in range(200):
#     observation, reward, done, info = env.step([0., 1, 0])
# print(time.time() - t)
#
#
# t = time.time()
# env.reset()
# for i in range(200):
#     observation, reward, done, info = env.step([0., 1, 0])
# print(time.time() - t)
#
