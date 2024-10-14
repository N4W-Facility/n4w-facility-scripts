import pandas as pd
from dotenv import load_dotenv, dotenv_values
#from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
import pykobo
import json
import requests
import io
import urllib.request

config_data = []
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

URL_KOBO = config_data['kobo_api_url']
API_VERSION = config_data['kobo_api_version']
KOBO_TOKEN = config_data['kobo_token']
USERNAME = config_data['kobo_username']
IMPLEMENTATION_ACTIVITY_FORM_UID = config_data["kobo_impl_activity_form_uid"]
DATABRICKS_IMPL_ACTIVITY_UNPROCESSED_FILE_PATH = config_data["databricks_impl_activity_unprocessed_path"]

#Kobo Methods
def fetch_kobo_data(url, token, api_version, form_uid): 
    # Initialize the Manager object
    km = pykobo.Manager(url=url, api_version=api_version, token=token)

    #Grab desired form
    try:
        kobo_form = km.get_form(form_uid)
    except:
        print("unable to get_form")
    print(kobo_form.metadata)

    #Get data associated with form
    try:
        kobo_form.fetch_data()
    except KeyError:
        print("Key Error warning - ignore")
    except:
        print("unable to fetch_data")
    return kobo_form.data

def delete_kobo_data(url, token): 
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.delete(url, headers=headers)
        json_response = json.loads(response.text)
    except:
        print("Unable to delete kobo data. Please do so manually.")

def fetch_kobo_media_files(url, token):
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.get(url, headers=headers)
        json_response = json.loads(response.text)
        media_files = [file for file in json_response['results'] if file['file_type'] == 'form_media' and file['metadata']['mimetype']=='text/csv']
    except:
        print("Unable to fetch media files")
    return media_files

def fetch_kobo_media_content(url, token):
    media = pd.DataFrame()
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.get(url, headers=headers)
        media_existing = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        media = pd.concat([media, media_existing], ignore_index=True)
    except:
        print("Unable to fetch media")
    return media

def delete_kobo_media_file(url, token, existing_file_id):
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.delete(f'{url}/{existing_file_id}', headers=headers)
    except:
        print("Unable to delete file")

def upload_kobo_media_file(url, token, new_data, filename): 
    #binary encode new file contents
    temp_data = f'./{filename}.csv'
    new_data.to_csv(temp_data)
    metadata = {
        'filename': f'{filename}.csv'
        ,'mimetype': 'text/csv'
    }
    data = {
        'user': '{URL_KOBO}api/v2/users/{USERNAME}/'
        ,'asset': url
        ,'description': filename
        #,'content': binary_data
        ,'file_type': 'form_media'
        ,'metadata': json.dumps(metadata)
    }
    file = {
        'content': open(temp_data,'rb')
    }
    #Upload file to asset (form) media library
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.post(url=f'{url}/files.json', headers=headers, data = data, files=file)
    except:
        print("Unable to upload new file")

def redeploy_kobo_form(asset_url, token):
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Token {token}'
    }
    version_to_deploy=''
    try:
        response = requests.get(asset_url, headers=headers, params={'format': 'json'})
        response_json = response.json()
        version_to_deploy = response.json()['version_id']
    except:
        print("couldn't fetch current form version")

    deployment_data = {
        'version_id': version_to_deploy,
        'active': True
    }
    try:
        response = requests.patch(asset_url + '/deployment/', headers=headers, data=deployment_data)
    except:
        print("could not redeploy form")

def merge_polygon_choices(new_choice_data, existing_choice_data, token):
    # Loop through new data and create new csv entries
    new_polygons = pd.DataFrame(columns=['impl_activity_program','impl_activity_polygon','impl_activity_plan','impl_activity_full_polygon'],data=None)
    for index, row in new_choice_data.iterrows():
        impl_activity_program = row['impl_activity_program']
        impl_activity_polygon = row['impl_activity_polygon']
        impl_activity_plan = row['impl_activity_plan']
        impl_activity_full_polygon = row['impl_activity_full_polygon']

        # If no existing implementation plan, assume new polygon
        if(impl_activity_plan == 0):
            new_polygons.loc[len(new_polygons.index)] = {
                'id': len(new_polygons.index),
                'program': impl_activity_program,
                'implementation_polygon': impl_activity_polygon,
                'implementation_plan': 0
            }

        # If marked as not full polygon, create new subdivision (increase letter)
        if(impl_activity_plan == 1 and impl_activity_full_polygon == 'no'):
            existing_max_subdivision = max(existing_choice_data[['implementation_polygon'].startswith(impl_activity_polygon)])
            existing_max_suffix = existing_max_subdivision.replace(impl_activity_polygon,"")
            # If no existing subdivisions, add "A" to end. Else implement letter by 1
            # NOTE: no more than 26 divisions at each "level" allowed!
            new_max_suffix = "A" if existing_max_suffix == "" else chr(ord(existing_max_suffix) + 1)
            new_max_subdivision = impl_activity_polygon + new_max_suffix
            new_polygons.loc[len(new_polygons.index)] = {
                'id': len(new_polygons.index),
                'program': impl_activity_program,
                'implementation_polygon': new_max_subdivision,
                'implementation_plan': impl_activity_plan
            }
            # Also download provided shapefile
            try:
                headers = {'Authorization': f'Token {token}'}
                url = row['impl_activity_new_subpolygon_file_URL']
                request = urllib.Request(url)
                request.add_header(headers)   
                urllib.request.urlretrieve(url, f'./UnprocessedShapefiles/{new_max_subdivision}')
            except:
                print("Unable to download and save provided shapefile")
            

    #Merge new and old polygons, and delete any duplicates
    updated_polygons = pd.concat([existing_choice_data, new_polygons], ignore_index=True)
    updated_polygons.drop_duplicates(inplace=True)

    return updated_polygons

def main():
    print("Monthly Run")

    #1. Fetch Kobo Data & Media files 
    impl_activity_form_data = fetch_kobo_data(URL_KOBO,KOBO_TOKEN,API_VERSION,IMPLEMENTATION_ACTIVITY_FORM_UID)
    impl_activity_form_existing_csvs = fetch_kobo_media_files(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files', KOBO_TOKEN)
    existing_polygon_file =  impl_activity_form_existing_csvs[['metadata']['filename']=='programs_implementation_polygons.csv'][0]
    existing_polygon_file_id = existing_polygon_file['uid']

    #2. Update Kobo csv: polygon choices for implementation activity
    #2a. Fetch existing Kobo csv files content
    existing_polygons = fetch_kobo_media_content(existing_polygon_file['content'],KOBO_TOKEN)
    new_polygon_file_content = merge_polygon_choices(impl_activity_form_data,existing_polygons, KOBO_TOKEN)
    #2b. Delete old polygons csv file from Kobo media library
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,existing_polygon_file_id)    
    #2c. Upload new polygons csv file to Kobo media library
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,new_polygon_file_content,'programs_implementation_polygons')
    #2d. Redeploy Implementation Activity form
    redeploy_kobo_form(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN)

    #3. Create Excel Output (Implementation_Activity.xlsx)
    impl_activity_form_data.to_excel("Implementation_Activity.xlsx","Implementation Activity")

    #4. Upload Excel Output to Databricks Volume (unprocessed folder)
    databricks = WorkspaceClient()
    try:
        with open('./Implementation_Activity.xlsx', 'rb') as file:
            file_bytes = file.read()
            binary_data = io.BytesIO(file_bytes)
            databricks.files.upload(DATABRICKS_IMPL_ACTIVITY_UNPROCESSED_FILE_PATH, binary_data, overwrite = True)
    except:
        print("unable to upload file to Databricks volume")

    #6. Download shapefiles (keep in other file)

    #5. Run Relevant Databricks Jobs to update table data
    
    #6. Merge new shapefiles into existing using "UnionShapefiles" methods
    #7. Delete data in Kobo using "DeleteKoboData" methods

if __name__ == '__main__':
    main()