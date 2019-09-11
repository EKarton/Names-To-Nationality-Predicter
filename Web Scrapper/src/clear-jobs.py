from redis_queue import RedisQueue

def clear_jobs():
    queue = RedisQueue(name='jobs', namespace='queue', decode_responses=True)
    job_in_json = queue.wait_and_dequeue()

    while job_in_json is not None:
        job_in_json = queue.wait_and_dequeue()

if __name__ == "__main__":
    clear_jobs()

