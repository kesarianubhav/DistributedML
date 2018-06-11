from rq import Queue , Worker , Connection
from redis import Redis
import redis


if __name__ == '__main__':
    with Connection():
        q = Queue()
        Worker(q).work()
