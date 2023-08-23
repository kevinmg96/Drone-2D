"""
prueba stop/resume thread
"""

import threading
import time


def doit(*arg):
    t = threading.current_thread()
    while not arg[2].is_set():
        print("working on %s" % arg[0])
        time.sleep(1)
        arg[1].wait()

    print("Stopping as you wish.")


def main():
    ev = threading.Event()
    stop_ev = threading.Event()

    t = threading.Thread(target=doit, args=("task",ev,stop_ev))
    t.start()

    ev.clear()
    time.sleep(5)  
    print("s")
    ev.set()
    time.sleep(10) 
    stop_ev.set()
    print("fin proces")

if __name__ == "__main__":
    main()