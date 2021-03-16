import random

class Cache:

    def contain(self, key):
        pass

    def get(self, key):
        pass

    def put(self, key, data):
        pass

class FixedSizeCache(Cache):
    def __init__(self, size=5000):
        self.size = size
        self.cache = dict()

    def contain(self, key):
        return key in self.cache

    def get(self, key):
        return self.cache.get(key, None)

    def put(self, key, data):
        if key in self.cache:
            return

        if len(self.cache) < self.size:
            self.cache[key] = data
        else:
            self.cache.pop(random.choice(dict.keys()))
