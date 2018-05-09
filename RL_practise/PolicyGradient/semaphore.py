#!/usr/bin/python

import threading
import sys
import time

class semaphore:
    def __init__(self,n_sems = 1,n_values=None):
        self.n_sems = n_sems
        if n_sems <= 0:
            self.n_sems = 1
        self.n_values = n_values 
        if self.n_sems > 1 and not isinstance(n_values,list):
            print("The number of initial values is less than the number of semaphores.")
            sys.exit(-1)
        self.sems = self.initialize_sem() 
        print("%d semaphore has initialized!"%self.n_sems)

    def initialize_sem(self):
        sems = []
        if self.n_sems == 1 and self.n_values is None:
            sem = threading.Semaphore(value = 1)
            sems.append(sem)
            return sems
        for i in range(self.n_sems):
            sem = threading.Semaphore(value=self.n_values[i])
            sems.append(sem)
        return sems

    def P(self,index):
        if index < 0 or index >= self.n_sems:
            print("out of index[0-%d]"%(self.n_sems-1))
            sys.exit(-1)
        #P operate
        self.sems[index].acquire()

    def V(self,index):
        if index < 0 or index >= self.n_sems:
            print("out of index[0-%d]"%(self.n_sems-1))
            sys.exit(-1)
        self.sems[index].release()

class Produce_consumer:
    def __init__(self,n_producers = 1,n_consumers = 1,max_step = 10,synchronize = True):
        self.max_step = max_step 
        self.n_producers = n_producers
        self.n_consumers = n_consumers
        self.sems = None
        if synchronize:
            #create semaphore
            self.sems = self.create_semaphore()
        self.producers = self.create_threads(self.n_producers,self.produce)
        self.consumers = self.create_threads(self.n_consumers,self.consumer)
        print("producers and consumers have initialized!!")
    
    def create_semaphore(self):
        sem = semaphore(n_sems = 2,n_values =[self.n_producers,0])
        return sem

    def create_threads(self,n_threads,target_fun):
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=target_fun)
            t.setDaemon(True)
            threads.append(t)
        return threads

    def start_threads(self,threads):
        for t in threads:
            t.start()

    def stop_threads(self,threads):
        for t in threads:
            #t.stop()
            t._Thread__stop()

    def join_threads(self):
        for t in self.producers:
            t.join()
        for t in self.consumers:
            t.join()
        print("all threads have exited!!")
        
    def start_produce_consumer(self):
        self.start_threads(self.producers)
        self.start_threads(self.consumers)
        print("producers and consumers have started!!")
        return

    def produce(self):
        while True:
            if self.max_step <= 0:
                print("stopping all threads....")
                self.stop_threads(self.producers)
                self.stop_threads(self.consumers)
                print("producers&consumers exits!!")
                break
            time.sleep(1)
            self.sems.P(index = 0)
            self.max_step -= 1
            print("producer[%s] has produced! Waiting for consumer....[%d]"%(threading.currentThread().name,self.max_step))
            self.sems.V(index = 1)
    
    def consumer(self):
        while True:
            time.sleep(1)
            self.sems.P(index = 1)
            print("comsumer[%s] has consumered! waiting for producer...."%threading.currentThread().name)
            self.sems.V(index = 0)

def main():
    a = threading.Event()
    pc = Produce_consumer(n_consumers=5,n_producers=4)
    pc.start_produce_consumer()
    pc.join_threads()
    print("main thread has exited!!")

if __name__ == "__main__":
    main()
