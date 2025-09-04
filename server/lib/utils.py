import random

def random_name_generator():
    adjectives = ["moist", "dank", "fluffy", "discombulated", "foreign", 
                  "sweet", "voluptuous", "thick", "pure", "used"]
    adjective = random.choice(adjectives)
    
    nouns = ["lemon", "pineapple", "orange", "apple", "grape"
             "banana", "tomato", "strawberry", "mango", "blueberry"]
    noun = random.choice(nouns)

    number = random.randint(100, 999)

    return adjective + noun.capitalize() + str(number)