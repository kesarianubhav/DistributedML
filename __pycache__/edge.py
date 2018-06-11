from packages import *
from node import Node

class Edge(object):

    __end1__ = None
    __end2__ = None
    __status__ = None

    def __init__(self,end1,end2):
        assert (isinstance(end1,Node) and isinstance(end2 ,Node)) , "Type Error"
        print("Edge Created ..")
        self.__end1__ = end1
        self.__end2__ = end2
        self.__end1__.add_out()
        self.__end2__.add_in()


    def get_end_status ( self ):
        a = self.__end1__
        b = self.__end2__
        a = a.get_status()
        b = b.get_status()
        return (a,b)


    def flow(self):
        (a,b)=self.get_end_status()
        # assert not ( (a=='l' and b=='l') or( a=='l' and b=='c')  ) , "Value Error"
        v1 =self.__end1__.get_value()
        self.__end2__.set_value(v1)



    def update_status(self):
        (a,b)=self.get_end_status()
        if a=='c' and b!='c':
            status = 1
        else:
            status= -1
        return status



if __name__ =="__main__":
    a =Node('scaler')
    print(a.get_status())
    a.set_value(8)
    print(a.get_status())
    b=Node('scaler')
    c=Node('+')
    e1 = Edge(a,c)
    e2 = Edge(b,c)
    e1.flow()
    e2.flow()
    print(c.get_status())
