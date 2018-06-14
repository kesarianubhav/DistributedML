import uuid
import os
# from keras.models import Sequential
# from keras.layers import Dense
import numpy as np
import math
import redis
from redis import Redis
from rq import Queue , Connection , Worker
from flask import jsonify
from qr import PriorityQueue
import json

os.system('sudo service redis start')

PORT = 6379
HOST = 'localhost'
prq =PriorityQueue('q',host=HOST,port=PORT)

"""


1. Model Creator
2. Calculation Distributor
3. Graph Creator
"""
def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


class Node(object):
    def __init__(self):
        self.inputs=None
        self.outputs=None
        
