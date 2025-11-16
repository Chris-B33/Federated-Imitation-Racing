import random


NOUNS       = ["lemon", "pineapple", "orange", "apple", "grape",
                "banana", "tomato", "strawberry", "mango", "blueberry"]
ADJECTIVES  = ["moist", "dank", "fluffy", "discombulated", "foreign", 
                "sweet", "voluptuous", "thick", "pure", "used"]


def generate_random_name():
    return ''.join([random.choice(ADJECTIVES), random.choice(NOUNS), str(random.randint(100, 999))])