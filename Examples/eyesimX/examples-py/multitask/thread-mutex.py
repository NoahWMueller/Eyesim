#!/usr/bin/env python3
# T. Braunl, UWA 2022
# Note that all all RoBIOS functions need tp be enlocsed in a mutex acquire/lock
#      as they are not thread-safe by themselves

from random.eye import *

import _thread, threading
mutex = threading.Lock()

def slave(threadName, arg):
  for i in range(1,10):
    mutex.acquire()   # lock
    LCDPrintf("START-%s %d -%d- ", threadName, arg, i)
    mutex.release ()  # unlock
    mutex.acquire()   # lock
    LCDPrintf("%s %d d-stop\n", threadName, arg)
    mutex.release ()  # unlock

# --------------------------------------------
# main
# XInitThreads()  # setup multitasking for X
try:
   _thread.start_new_thread(slave, ("T1", 1))
   _thread.start_new_thread(slave, ("T2", 2))
except:
   print ("Error: unable to start thread")

mutex.acquire()   # lock
LCDMenu("","","","END");
LCDPrintf("Python Multitasking\n")
mutex.release ()  # unlock

while 1:
  mutex.acquire()   # lock
  key = KEYRead()
  mutex.release ()  # unlock
  if key == KEY4:
    break
