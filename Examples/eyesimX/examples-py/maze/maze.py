#!/usr/bin/env python3
from random.eye import *

#include "eyebot.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define DIST     360
#define SPEED    180
#define ASPEED    45
#define THRES    175
#define MAZESIZE  16

DIST = 360
SPEED = 180
ASPEED = 45
THRES = 175
MAZESIZE = 16

''' pos.0,0 is bottom left, dir. 0 is facing up (north) '''
mark = [([0]*MAZESIZE) for i in range(MAZESIZE)]   # 1 if visited
#int wall[MAZESIZE+1][MAZESIZE+1][2]  ''' 1 if wall, 0 if free, -1 if unknown '''
wall = [0]*(MAZESIZE+1)
for i in range(len(wall)):
    wall[i] = [0]*(MAZESIZE+1)
for i in range(MAZESIZE+1):
    for j in range(MAZESIZE+1):
        wall[i][j] = [0]*2
'''  BOTTOM: wall[x][y][0]  LEFT: wall[x][y][1] '''
map = [([0]*MAZESIZE) for i in range(MAZESIZE)]     #distance to goal 
nmap = [([0]*MAZESIZE) for i in range(MAZESIZE)]    #copy
path = [0]*MAZESIZE*MAZESIZE    #shortest path

GRAPH=1
DEBUG=0
DEBUG2=0

'''* print mark full to X window (sim only).
    print positions that robot has already visited.  '''
def print_mark_W():
    ''' print to window (sim. only) '''
    print("MARK\n")
    for i in range(MAZESIZE-1,-1,-1):
        for j in range(0,MAZESIZE):
          if (mark[i][j]): print("x ")
          else: print(". ")
        print("\n")
    print("\n")

'''* print mark on LCD (6 lines).
    print positions that robot has already visited.  '''
def print_mark():
    LCDSetPos(1,0)
    for i in range(5,-1,-1):
        for j in range(0,14):
            if (mark[i][j]): LCDPrintf("x")
            else:LCDPrintf(".")
        if (i>0):
            LCDPrintf("\n")

'''* print ful maze in X window (sim only).
    print maze as explored by robot.  '''
def print_maze_W():
    print("MAZE\n")
    for i in range(MAZESIZE,-1,-1):
        for j in range(0,MAZESIZE+1):
            if (wall[i][j][1]==1): print("|")  #left
            elif (wall[i][j][1]==0): print(" ")
            else: print(".")
            if (wall[i][j][0]==1): print("_")  # bottom
            elif (wall[i][j][0]==0): print(" ")
            else: print(".")
        print("\n")
    print("\n")

'''* print ful maze on LCD (6 lines).
    print maze as explored by robot.  '''
def print_maze():
    LCDSetPos(1,0)
    for i in range(5,-1,-1):
        for j in range(0,7):
            if (wall[i][j][1]==1): LCDPrintf("|")  #left 
            elif (wall[i][j][1]==0): LCDPrintf(" ")
            else: LCDPrintf(".")
            if (wall[i][j][0]==1): LCDPrintf("_")  #bottom 
            elif (wall[i][j][0]==0): LCDPrintf(" ")
            else: LCDPrintf(".")
        if i>0: LCDPrintf("\n")
  
def wall_set(wall,x,y,z, v):
    if wall[x][y][z] == -1:
        wall[x][y][z] = v  # not seen before, set value
    elif wall[x][y][z] != v:     #seen before and CONTRADITION 
        wall[x][y][z] = 1        #assume wall to be safe 
        LCDSetPrintf(0,0,"CONTRADICTION\n") #AUBeep()

'''* maze_entry.
    enter recognized walls or doors.  '''
def maze_entry(x, y, dir, open):
    if dir == 0:
        wall_set(wall, y+1, x, 0, int(not open)) # top = bottom of next 
    elif dir == 2:
        wall_set(wall, y, x, 0, int(not open))
    elif dir == 1:
        wall_set(wall, y, x, 1, int(not open))
    elif dir == 3:
        wall_set(wall, y, x+1, 1, int(not open)) #right = left of next

'''* robo position and orientation.
   dir = 0, 1, 2, 3 equals: north, west, south, east.  '''
rob_x = 0
rob_y = 0
rob_dir = 0

'''* init_maze.
    inits internal map of maze.
    set marks to 0 and walls to -1 (unknown).  '''
def init_maze():
    for i in range(MAZESIZE):
        for j in range(MAZESIZE):
            mark[i][j] = 0
    for i in range(MAZESIZE+1):
        for j in range(MAZESIZE+1):
            wall[i][j][0] = -1
            wall[i][j][1] = -1 

def xneighbor(x, dir):
    if dir == 0: return x #north 
    elif dir == 1: return x-1 #west  
    elif dir == 2: return x #south 
    elif dir == 3: return x+1 # east  
  
def yneighbor(y, dir):
    if dir == 0: return y+1 # north 
    elif dir == 1: return y # west  
    elif dir == 2: return y-1 # south 
    elif dir == 3: return y # east  
    

def unmarked(y, x, dir):
    dir = int((dir+4) % 4)
    return not mark [yneighbor(y,dir)] [xneighbor(x,dir)]



'''* go_to.
  walk one square in current direction '''
def go_to(dir):
    global rob_x, rob_y, rob_dir
    dir = (dir+4) % 4  # keep goal dir in 0..3 
    turn = dir - rob_dir
    if turn == 3:
        turn = -1  # turn shorter angle 
    elif turn == -3:
        turn =  1

    if turn:
        if DEBUG:
            LCDSetPrintf(13,0, "Turn %d %d   ", turn*90, ASPEED)
        VWTurn(turn*90, ASPEED)  # turn 
        VWWait()
  
    if DEBUG:
        LCDSetPrintf(13,0, "Straight %d %d   ", DIST, SPEED)
    VWStraight(DIST, SPEED)    # go one step 
    VWWait()

    cur_x, cur_y, cur_p = VWGetPosition()
    if DEBUG:
        LCDSetPrintf(14,0, "X %d Y %d Phi %d   ", cur_x, cur_y, cur_p)

    rob_dir = dir
    rob_x   = xneighbor(rob_x,rob_dir)
    rob_y   = yneighbor(rob_y,rob_dir)



'''* check_mark.
    if ALL walls of a square are known, mark square as visited.
    this avoids unnecessary exploration.  '''
def check_mark():
    for i in range(1,MAZESIZE):
        for j in range(0,MAZESIZE):
            ''' careful: watch boundaries!! i from 1 / j until size-1 '''
            if wall[i  ][j][0] != -1 and wall[i][j  ][1] != -1 \
            and wall[i+1][j][0] != -1 and wall[i][j+1][1] != -1: #bottom / left, top / right 
                mark[i][j] = 1
  


'''*  explore.
    search maze goal from given start position and orientation.
    if more than one possible way: search all recursively.
    mark all visited maze squares.  '''
def explore():
    global rob_x, rob_y, rob_dir
    mark[rob_y][rob_x] = 1   # mark current square 
    left_open  = int(PSDGet(PSD_LEFT) > THRES)
    front_open = int(PSDGet(PSD_FRONT) > THRES)
    right_open = int(PSDGet(PSD_RIGHT) > THRES)
    maze_entry(rob_x,rob_y,rob_dir,       front_open)
    maze_entry(rob_x,rob_y,(rob_dir+1)%4, left_open)
    maze_entry(rob_x,rob_y,(rob_dir+3)%4, right_open)
    check_mark()
    old_dir = rob_dir

    if GRAPH:
        LCDSetPos(0,0)
        LCDPrintf("Pos[%2d,%2d,%1d]", rob_x,rob_y,rob_dir)
        if left_open: LCDSetPrintf(0,13,"<")
        else: LCDSetPrintf(0,13,"|")
        if front_open: LCDSetPrintf(0,14,"^")
        else: LCDSetPrintf(0,14,"-")
        if right_open: LCDSetPrintf(0,15,">")
        else: LCDSetPrintf(0,15,"|")
        print_maze()


    if DEBUG:
        print_mark_W()
        print_maze_W()
        LCDMenu("Next"," "," "," ")
        KEYWait(KEY1)
        LCDMenu(" "," "," "," ")

    if front_open and unmarked(rob_y,rob_x,old_dir):   # then go straight 
        go_to(old_dir)   # go 1 forward, 0 if first choice 
        explore()        # recursive call 
        go_to(old_dir+2) # go 1 back 

    if left_open and unmarked(rob_y,rob_x,old_dir+1):  # then turn left 
        go_to(old_dir+1) # go 1 left 
        explore()        # recursive call 
        go_to(old_dir-1) # go 1 right, -1 = +3 

    if right_open and unmarked(rob_y,rob_x,old_dir-1): # then turn right 
        go_to(old_dir-1) # go 1 right, -1 = +3 
        explore()        # recursive call 
        go_to(old_dir+1) # go 1 left 
    

'''* print shortest distances from start in X window (sim only).  '''
def print_map_W():
    print("MAP\n")
    for i in range(MAZESIZE-1,-1,-1):
        for j in range(MAZESIZE):
            print("%3d",map[i][j])
        print("\n")
    print("\n")


'''* print shortest distances from start on LCD (6 lines).  '''
def print_map():
    LCDClear()
    LCDPrintf("Map distances\n")
    for i in range(5,-1,-1):
        for j in range(4):
          LCDPrintf("%3d",map[i][j])
        if i>0: LCDPrintf("\n")
  

'''* shortest_path.
    analyze shortest path after maze has been searched.
    returns path length to goal.
    or -1 if no path could be found.  '''
def shortest_path(goal_y, goal_x):
    LCDSetPrintf(0,0, "                ") #clear top line 
    LCDSetPrintf(0,0, "Map..")
    for i in range(MAZESIZE):
        for j in range(MAZESIZE):
            map [i][j] = -1  #init
            nmap[i][j] = -1
            
    map [0][0] = 0
    nmap[0][0] = 0
    iter=0

    while True:
        iter += 1
        for i in range(MAZESIZE):
            for j in range(MAZESIZE):
                if map[i][j] == -1:
                    if i>0:
                        if not wall[i][j][0] and map[i-1][j] != -1:
                            nmap[i][j] = map[i-1][j] + 1
                    if i<MAZESIZE-1:
                        if not wall[i+1][j][0] and map[i+1][j] != -1:
                            nmap[i][j] = map[i+1][j] + 1
                    if j>0:
                        if not wall[i][j][1] and map[i][j-1] != -1:
                            nmap[i][j] = map[i][j-1] + 1
                    if j<MAZESIZE-1:
                        if not wall[i][j+1][1] and map[i][j+1] != -1:
                            nmap[i][j] = map[i][j+1] + 1

        for i in range(MAZESIZE):
            for j in range(MAZESIZE):
                map[i][j] = nmap[i][j]  # copy back 

        if DEBUG2:
            print_map()
            print_map_W()
            LCDMenu("Next"," "," "," ")
            KEYWait(KEY1)
            LCDMenu(" "," "," "," ")
        if map[goal_y][goal_x] != -1 or iter >= (MAZESIZE*MAZESIZE): break
    
    LCDPrintf("done\n")
    return map[goal_y][goal_x]



'''* build path.
  build shortest path after finding it.
  uses map and wall.
  sets path.  '''
def build_path(i, j, len):
    LCDSetPrintf(0,0, "                ") # clear top line 
    LCDSetPrintf(0,0, "Path..")
    
    if i<=5 and j<=6:
        LCDSetPrintf(6-i,2*j+1, "G") # mark goal 
    for k in range(len-1,-1,-1):
        if i>0 and not wall[i][j][0] and map[i-1][j] == k:
            i -= 1
            path[k] = 0 # north 
        elif i<MAZESIZE-1 and not wall[i+1][j][0] and map[i+1][j] == k:
            i += 1
            path[k] = 2 # south 

        elif j>0 and not wall[i][j][1] and map[i][j-1] == k:
            j -= 1
            path[k] = 3 # east 

        elif j<MAZESIZE-1 and not wall[i][j+1][1] and map[i][j+1] == k:
            j += 1
            path[k] = 1 # west 
        else:
            LCDPrintf("ERROR") #AUBeep()
            KEYWait(ANYKEY)
            
        ''' mark path in maze on LCD '''
        if i<=5 and j<=6:
            if k>0:
                LCDSetPrintf(6-i,2*j+1, "*") # path 
            else:
                LCDSetPrintf(6-i,2*j+1, "S") # start                        
        if DEBUG2:
            print("path %3d:%1d\n", k, path[k])

    LCDSetPrintf(0,6, "done")



'''* drive path.
  drive path after building it.
  parametere specifies start to finish (0).
  or finish to start (1).  '''
def drive_path(len, reverse):
    global rob_x, rob_y, rob_dir
    if reverse:
        for i in range(len-1,-1,-1):
            go_to(path[i]+2)

        if rob_dir != 0: # back in start field 
            VWTurn(-rob_dir*90, ASPEED)  # turn 
            VWWait()
            rob_dir = 0
    else:
        for i in range(len):
            go_to(path[i])



'''* main program.
    search maze from start in (0,0).
    search whole maze and generate map.
    @AUTHOR    Thomas Braunl, UWA, 1998.-- Updated 2017  '''
def main():
    global rob_x, rob_y, rob_dir
    LCDPrintf("MAZE\nfull search\n\n")
    init_maze()

    LCDMenu("GO","ROT","DEBUG","END")
    key = KEYGet()
    DEBUG = int(key == KEY3)
    if key == KEY2:
        VWTurn(90, 90)
        VWWait()
    if key == KEY4:
        return 0

    rob_x = 0
    rob_y = 0
    rob_dir = 0  # start in pos. 0,0, facing north
    explore()

    ''' back in [0,0] turn robot to original direction '''
    if rob_dir != 0:
        VWTurn( -rob_dir*90, ASPEED)  # turn 
        VWWait()
        rob_dir = 0

    while True:
        print_maze()
        LCDMenu("Y+-","X+-","+/-","GO")
        incr = 1
        goalX=0
        goalY=0
        while True:
            LCDSetPos(0,0)
            LCDPrintf("GOAL y,x: %2d %2d", goalY, goalX) #AUBeep()
            if goalY<=5 and goalX <=6:
                LCDSetPos(6-goalY,2*goalX+1)
            key = KEYGet()
            if key == KEY1:
                goalY = (goalY+incr+MAZESIZE) % MAZESIZE
                break
            elif key == KEY2:
                goalX = (goalX+incr+MAZESIZE) % MAZESIZE
                break
            elif key == KEY3:
                incr = -incr
                break
            elif key == KEY4:
                break
            
        LCDMenu(" "," "," "," ")
        path_len = shortest_path(goalY,goalX)  # from start 0,0 to goal 
        if path_len != -1: # if path exists 
            print_map_W()
            LCDSetPos(0,0)
            LCDPrintf("Path length: %3d", path_len)
            build_path(goalY,goalX,path_len)

            while True:  # optional data display 
                LCDMenu("Map","Mrk","Maz","DRV")
                key = KEYGet()
                if key == KEY1:
                    print_map()
                    break
                elif key == KEY2:
                    print_mark()
                    break
                elif key == KEY3:
                    print_maze()
                    break
                elif key == KEY4:
                    break
            drive_path(path_len,0) # drive start to finish 
            drive_path(path_len,1) # drive finish to start 
        else:
            LCDSetPos(0,0)
            LCDPrintf("No path exists!") #AUBeep() 

        LCDMenu("REP"," "," ","END")
        if KEYGet() == KEY4: break
    return 0


if __name__ == "__main__":
    main()
