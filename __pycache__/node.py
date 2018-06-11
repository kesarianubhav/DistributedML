from packages import *
import tinydb as tdb
from tinydb import TinyDB , Query
from tinydb import where
import ast

class Node(object):

    __status__ = ['l','w','c']
    __node_type__ = ['+','*','-','scaler','vector']

    __current_status__ = None
    __value__ = None

    __out__ = 1
    __in__ = 0
    #__completed_nodes__ =0

    def __init__(self,node_type_name):
        #constructor for the node
        global db
        assert node_type_name in self.__node_type__ ,  ("Type Eror ")

        self.__current_status__='l'
        self.__node_type__=node_type_name

        db=TinyDB('db_'+str(node_type_name)+'.json')
        db.insert({"status":self.__current_status__})
        db.insert({"in":self.__in__})
        db.insert({"out":self.__out__})
        db.insert({"node_type":self.__node_type__})
        db.insert({"value":self.__value__})
        # print(db.all())

    def set_status(self):
        #sets the current status of the node
        # assert isinstance(arg,numpy.ndarray) , ("Type Error")
        if(self.__node_type__ == 'vector' or self.__node_type__=='scaler'):
            if self.__current_status__ == 'l':
                self.__current_status__ =='c'

        else:
            if self.__current_status__ == 'l':
                self.__current_status__ =='w'
            if self.__current_status__ == 'w':
                self.__current_status__ =='c'

    def get_status(self):
        #get the current status of the node
        return self.__current_status__

    def get_node_type(self):
        #get the node_type in which Node is created
        return self.__node_type__

    def add_in(self):
        self.__in__+=1

    def add_out(self):
        self.__out__+=1

    def set_value(self,arg1):
        # self.lock.acquire()
        if self.__node_type__ == 'scaler' or self.__node_type__ == 'vector' :
            self.__value__=arg1
            self.set_status()


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


        self.__in__=self.__in__+1
        # User = Query()
        # a=((db.all()))
        # b=(ast.literal_eval(str(a[1])))
        # no = b['current_split']

        if ( self.__in__ == self.__out__ ):
            self.__current_status__ = 'c'
        else :
            self.__current_status__ = 'w'
        # self.lock.release()

    def get_value ( self ):
        return self.__value__



if __name__=="__main__":
    a =Node("scaler")
    print(a.get_node_type())
    print(a.get_status())
    a.set_value(6)
    print(a.get_status())
