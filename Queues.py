from redis import Redis
from rq import Queue
import redis

class MasterQueue(object):

    def __init__(self,host='localhost',port=9999):
        pool = redis.ConnectionPool(host=host, port=port, db=0)
        r=redis.Redis(connection_pool=pool)
        self.__mq__ = Queue(connection=r)
        print("Maseter Queue Running on Port "+str(port))

    def get_master_queue(self):
        return self.__mq__


class AppQueue(object):

        def __init__(self,host='localhost',port=8888):
            pool = redis.ConnectionPool(host=host, port=port, db=0)
            r=redis.Redis(connection_pool=pool)
            self.__q__ = Queue(connection=r)
            print("App Queue Running at Port "+str(port) )

        def get_app_queue(self):
            return self.__q__



if __name__=="__main__":
    print("Just to check")
    q1 = MasterQueue(9779)
    q2 = AppQueue(6379)
    print(q1)
    print(q2)
