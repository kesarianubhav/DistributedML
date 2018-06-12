from flask import Flask, jsonify, request
from redis import StrictRedis
from rq import Queue
from flask_redis import Redis


from random import randrange

# from settings import REDIS_HOST, REDIS_PORT
REDIS_HOST = 'localhost'
REDIS_PORT = 9999

app = Flask(__app__)
app.config['REDIS_HOST'] = REDIS_HOST
app.config['REDIS_PORT'] = REDIS_PORT
app.config['REDIS_DB'] = 0
redis1 = Redis(app)



@app.route('/')
def get_randrange():
    if request is not None :
        print("I HAVE GOT THE REQUEST :"+str(request))
    else:
        print("KUCH NHI HAI MERE PAAS ")

    if 'stop' in request.args:
        #
        # stop = int(request.args.get('stop'))
        # start = int(request.args.get('start', 0))
        # step = int(request.args.get('step', 1))
        #
        # job = q.enqueue(randrange, start, stop, step, result_ttl=5000)
        #
        # return jsonify(job_id=job.get_id())
        print(request.args)
        return jsonify(request.args)


    return 'Stop value not specified!', 400


@app.route("/results")
@app.route("/results/<string:job_id>")
def get_results(job_id=None):

    if job_id is None:
        return jsonify(queued_job_ids=q.job_ids)

    job = q.fetch_job(job_id)

    if job.is_failed:
        return 'Job has failed!', 400

    if job.is_finished:
        return jsonify(result=job.result)

    return 'Job has not finished!', 202

if __name__ == '__main__':
    # Start server
    app.run(host='127.0.0.1', port=9999, debug=True)
