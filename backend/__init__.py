import os

from dotenv import load_dotenv

def get_secret_key(key):
    response = load_dotenv('backend/secret.env')
    if not response:
        raise Exception('Failed load secret')
    return os.environ.get(key)