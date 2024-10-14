import pandas as pd
import json
import requests

config_data = []
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

URL_KOBO = config_data['kobo_api_url']
API_VERSION = config_data['kobo_api_version']
KOBO_TOKEN = config_data['kobo_token']
USERNAME = config_data['kobo_username']
IMPLEMENTATION_ACTIVITY_FORM_UID = config_data["kobo_impl_activity_form_uid"]
DATABRICKS_IMPL_ACTIVITY_UNPROCESSED_FILE_PATH = config_data["databricks_impl_activity_unprocessed_path"]

def delete_kobo_data(url, token): 
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.delete(url, headers=headers)
        json_response = json.loads(response.text)
    except:
        print("Unable to delete kobo data. Please do so manually.")

def main():
    print("Clear Kobo Data")
    delete_kobo_data(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/data',KOBO_TOKEN)   

if __name__ == '__main__':
    main()