from packages import *
import tinydb as tdb
from tinydb import Tinydb , Query



class Node(object):

    __status__ = ['l','w','c']
    __node_type__ = ['+','*','-','scaler','vector']

    __current_status__ = None
    __value__ = None

    __max_split__ = 10
    __current_split__ = 0


    def __init__(self,node_type_name):
        #constructor for the node
        assert node_type_name in self.__node_type__ ,  ("Type Eror ")
        self.__current_status__='l'
        self.__node_type__=node_type_name


    def set_status(self,arg):
        #sets the current status of the node
        if(self.__node_type__ == 'vector'):
            assert isinstance(arg,numpy.ndarray) , ("Type Error")
        assert arg in status , "Wrong Status Code "
        self.__current_status__ = arg

    def get_status(self):
        #get the current status of the node
        return self.__current_status__

    def get_node_type(self):
        #get the node_type in which Node is created
        return self.__node_type__

    def set_value(self,arg1):
        # self.lock.acquire()
        if self.__node_type__ == 'scaler' or self.__node_type__ == 'vector' :
            self.__value__=arg1

        elif self.__node_type__=='+':
            if self.__value__ == None:
                self.__value__=arg1


            else:
                assert ( type(arg1) == type(self.__value__)) , "Type Error"
                if isinstance(arg1,np.ndarray):
                    self.__value__ = np.add(arg1 , self.__value__ )
                else:
                    self.__value__= self.__value__ + arg1

        elif self.__node_type__=='-':
            if self.__value__ == None:
                self.__value__=arg1

            else:
                assert ( type(arg1) == type(self.__value__)) , "Type Error"
                if isinstance(arg1,np.ndarray):
                    self.__value__ = np.subtract(arg1 , self.__value__)
                else:
                    self.__value__=self.__value__ - arg1

        elif self.__node_type__=='*':
            if self.__value__ == None:
                self.__value__=arg1

            else:
                assert ( type(arg1) == type(self.__value__)) , "Type Error"
                if isinstance(arg1,np.ndarray):
                    self.__value__ = np.dot(arg1 , self.__value__)
                else:
                    self.__value__=self.__value__ * arg1



        #global current_status
        self.__current_status__ = 'c'
        # self.lock.release()

    def get_value ( self ):
        return self.__value__




if __name__=="__main__":
    a =Node("scaler")
    print(a.get_node_type())
    print(a.get_status())
    a.set_value(6)
    print(a.get_status())

    # a.__change_node__()
