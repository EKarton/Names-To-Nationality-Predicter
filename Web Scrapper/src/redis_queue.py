import redis

class RedisQueue:
	def __init__(self, name, namespace='queue', **redis_kwargs):
		self._db= redis.Redis(**redis_kwargs)
		self.key = '%s:%s' %(namespace, name)

	'''
		Returns the size of the queue
	'''
	def size(self):
		return self._db.llen(self.key)

	'''
		Determines if the queue is empty
	'''
	def is_empty(self):
		return self.size() == 0

	'''
		Puts a job to the back of the queue
	'''
	def enqueue(self, item):
		self._db.rpush(self.key, item)

	''' 
		This gets the next available job in the queue.
		If there are no jobs available in the queue, it will wait for {@code timeout} ms.
		If {@code timeout} is not set, it will wait infinitely.
		If the wait time has timed out, it will return None.
	'''
	def wait_and_dequeue(self, timeout=None):
		result = self._db.blpop(self.key, timeout=timeout)

		print(result)
		item = None
		if result:
			item = result[1]
		return item

	'''
		This gets the next available job in the queue.
		If there are no jobs available, it will return None.
	'''
	def dequeue(self):
		result = self._db.lpop(self.key)

		item = None
		if result:
			item = result[1]
		return item