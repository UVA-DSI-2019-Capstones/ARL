import os
try:
    os.system('python ./setup.py')
    print('database created')
except:
    print("couldn't setup database")

try:
    os.system('python ./seed.py')
    print('avatar prompts seeded')
except:
    print("couldn't seed avatar prompts")

try:
    os.system('python ./seed_trainee_response.py')
    print('trainee prompts seeded')
except:
    print("couldn't seed trainee responses")

try:
    os.system('python ./seed_turk_response.py')
    print('turk prompts seeded')
except:
    print("couldn't seed turk responses")

try:
    os.system('python ./seed_context_data.py')
    print('context data seeded')
except:
    print("couldn't seed context data")