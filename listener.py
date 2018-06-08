import socket
import sys
import os

class Listener(object):

    def __init__(self,port=8888):
        self.__socket__=socket.socket()
        self.__port__=port
        self.__socket__.bind(('',port))
        print ("socket binded to %s" %(port))

        self.__socket__ . listen(10)


    def listen(self):
        while(1):
            c,addr = self.__socket__.accept()
            print("got request from "+str(addr))
            c.send(b"Connected with you")
            c.close()


if __name__ == "__main__":
    l = Listener(8888)
    l.listen()
