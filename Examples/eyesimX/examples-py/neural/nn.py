#!/usr/bin/env python3
from random.eye import *
import math
import random

NUM_TRAINING = 200
NUM_TEST = 100

LAYERS = 3
LN = LAYERS - 1
NEURONS = 257

SPA = 24
OFF = 192

_max = []

act = []
out = []
w = []

class NN():
    def __init__(self):
        self.input = []
        self.solution = []
        self.sol = None

Images = []

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def feed_forward(N_in):
    global act, out
    for i in range(_max[0]):
        out[0][i] = N_in[i]
    out[0][_max[0]] = 1

    for layer in range(1, LAYERS):
        for i in range(_max[layer]):
            act[layer][i] = 0
            for j in range(_max[layer - 1]):
                act[layer][i] += out[layer-1][j] * w[layer-1][j][i]
            out[layer][i] = sigmoid(act[layer][i])
        out[layer][_max[layer]] = 1

def back_prop(train_in, train_sol):
    global w
    # Run network, calculate difference to desired output
    feed_forward(train_in)
    err_total = 0
    err = [[0 * NEURONS] for _ in range(LAYERS)]
    diff = [[0 * NEURONS] for _ in range(LAYERS)]
    
    # A. Calculate output error and output Diff
    for i in range(_max[LN]):
        err[LN][i] = train_sol[i] - out[LN][i]
        err_total += err[LN][i] * err[LN][i]
        diff[LN][i] = err[LN][i] * (1 - out[LN][i]) * out[LN][i]

    # B. Work backwards through all layers
    for layer in range(LAYERS-2, 0, -1):
        for i in range(_max[layer]):
            diff[layer][i] = 0
            for j in range(_max[layer+1]):
                diff[layer][i] += diff[layer+1][j] * w[layer][i][j] * (1-out[layer][i]) * out[layer][i]
                w[layer][i][j] += diff[layer+1][j] * out[layer][i]
    return err_total

def init_weights():
    global w
    for layer in range(LAYERS-1):
        for i in range(_max[layer]):
            for j in range(_max[layer+1]):
                w[layer][i][j] = random.random() / ( RAND_MAX * _max[layer] )

def testing(num):
    display16x16(Images[num].input)
    feed_forward(Images[num].input)
    LCDSetPrintf(11,30, "0   1   2   3   4   5   6   7   8   9")
    LCDSetPrintf(14,30, "Pattern " + str(num) + " (" + str(Images[num].sol) + ")")
    if num < NUM_TRAINING:
        LCDSetPrintf(14,47, "TRAINING")
    else:
        LCDSetPrintf(14,47, "UNKNOWN")
    LCDSetPos(12,30)
    for i in range(10):
        LCDPrintf(str(out[LAYERS-1][i]))
        LCDLine(SPA*i+OFF, 140, SPA*i+OFF, 0, BLACK)
        LCDLine(SPA*i+OFF, 140, SPA*i+OFF, 140 - int(75*(out[LAYERS-1][i])), RED)

def training():
    for i in range(NUM_TRAINING):
        train_err = back_prop(Images[i].input, Images[i].solution)
        train_err += train_err
    LCDSetPrintf(14, 0, "Err = " + str(total_err))

def loadPGM












def display16x16(matrix):
    pass