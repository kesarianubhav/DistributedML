from Queues import AppQueue
from rq import Connection
import numpy as np
from time import time
import time
import os
from rq import Queue,Connection
from function import task

class App(object):

    def __init__(self,name,operation):
        print("App "+str(name)+" created \n")
        self.__operation__ = operation
        self.__value__=0
        self.__status__='l'
        self.__appname__=name
        self.__arg__=[None]*2



    def get_status(self):
        return self.__status__

    def get_value(self):
        return self. __value__

    def set_arg(self,arg1,arg2):
        self.__arg__[0]=arg1
        self.__arg__[1]=arg2

    def task_fn(self,arg1,arg2):
        if self.__operation__ == '*':
            p = arg1 * arg2
        if self.__operation__ == '+':
            p = arg1+arg2
        if self.__operation__=='-':
            p = arg1 - arg2
        if self.__operation__=='dot':
            p = np.dot(arg1 , arg2)
        self.__value__=p


    def activate(self,worker=3):
        #print("bhk yha se")
        #print(self.__arg__[0])
        #
        # a = AppQueue(port=6379)
        # q = a.get_app_queue()

        #self.__q__ = q

        chunks = np.array_split( self.__arg__[0],worker,axis =0)
        #
        # with Connection():

        async_results={}
        q=Queue()
        for i in range(0,worker):
            async_results[i] = q.enqueue( task, args=(chunks[i] , self.__arg__[1]) )
        done = False

        start_time = time.time()

        while not done :
            os.system('clear')
            print('App Waiting Time : (now = %.2f)' % (time.time() - start_time,))
            done = True
            for x in range(0,worker):
                result = async_results[x].return_value
                print(result)
                if result is None:
                    done = False
                    result = '(waiting)'
                print('chunk(%d) = %s' % (x, result))
                # np.vstack((self.__value__,result))
            time.sleep(0.2)

        p=async_results[0].return_value

        for i in range (1,worker):
            p=np.vstack((p,async_results[i].return_value))



        # if q.empty():
        self.__value__=p
        #os.system('clear')
        self.__value__ = (p)
        self.__status__='c'
        print('App Completed with Result of shape'+str(self.__value__.shape))
