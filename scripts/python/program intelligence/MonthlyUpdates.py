import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv, dotenv_values
from databricks.sdk import WorkspaceClient
import json
import io
import urllib.request
from apiMethods.KoboInputs import fetch_kobo_data,fetch_kobo_media_files,fetch_kobo_media_content,delete_kobo_media_file,upload_kobo_media_file,redeploy_kobo_form
#import pykobo
#import requests
#from databricks.connect import DatabricksSession

config_data = []
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

URL_KOBO = config_data['kobo_api_url']
API_VERSION = config_data['kobo_api_version']
KOBO_TOKEN = config_data['kobo_token']
USERNAME = config_data['kobo_username']
IMPLEMENTATION_ACTIVITY_FORM_UID = config_data['kobo_implementation_activity_form_uid']
DATABRICKS_IMPL_ACTIVITY_UNPROCESSED_FILE_PATH = config_data['databricks_implementation_activity_unprocessed_path']

def merge_polygon_choices(new_choice_data, existing_choice_data, token):
    # Create Dataframe to store new polygon records
    new_polygons = pd.DataFrame(columns=['program','polygon','plan_exists','full_polygon'],data=None)
    # Loop through new data and create new polygon records 
    for index, row in new_choice_data.iterrows():
        impl_activity_program = row['program']
        impl_activity_polygon = row['polygon']
        impl_activity_plan = row['plan_exists']
        impl_activity_full_polygon = row['full_polygon']

        # If no existing implementation plan, assume new polygon?
        #if(plan_exists == 0):
        #    new_polygons.loc[len(new_polygons.index)] = {
        #        'id': len(new_polygons.index),
        #        'program': program,
        #        'implementation_polygon': polygon,
        #        'implementation_plan': 0
        #    }

        # If marked as not full polygon, create new subdivision (increase letter)
        if(impl_activity_plan == 1 and impl_activity_full_polygon == 'no'):
            # Check both in preexisting and polygons, and new polygons created during this program run
            existing_polygon_subdivisions = existing_choice_data[['implementation_polygon'].startswith(impl_activity_polygon)]['implementation_polygon'] \
                + new_polygons.startswith(impl_activity_polygon)['implementation_polygon']
            existing_max_subdivision = max(existing_polygon_subdivisions)
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

def merge_other_choices(new_data, choice_file, choice_file_fieldnames):
    existing_data = fetch_kobo_media_content(choice_file['content'],KOBO_TOKEN)
    updated_data = pd.concat([existing_data[choice_file_fieldnames], new_data], ignore_index=True)
    updated_data.drop_duplicates(inplace=True)
    updated_data.replace('', np.nan, inplace=True)
    updated_data.dropna(inplace=True)
    return updated_data

def main():
    print("Monthly Run")

    #1. Fetch Kobo Data & Media files 
    impl_activity_form_data = fetch_kobo_data(URL_KOBO,KOBO_TOKEN,API_VERSION,IMPLEMENTATION_ACTIVITY_FORM_UID)
    impl_activity_form_existing_csvs = fetch_kobo_media_files(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files', KOBO_TOKEN)
    existing_polygon_file =  list(filter(lambda x: x['metadata']['filename'] == 'programs_implementation_polygons.csv', impl_activity_form_existing_csvs))[0]
    existing_polygon_file_id = existing_polygon_file['uid']

    #2. Update Kobo csv: polygon choices for implementation activity
    #2a. Fetch existing Kobo csv files content
    existing_polygons = fetch_kobo_media_content(existing_polygon_file['content'],KOBO_TOKEN)
    new_polygon_file_content = merge_polygon_choices(impl_activity_form_data,existing_polygons, KOBO_TOKEN)
    #2b. Delete old polygons csv file from Kobo media library
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,existing_polygon_file_id)    
    #2c. Upload new polygons csv file to Kobo media library
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,new_polygon_file_content,'programs_implementation_polygons')
    
    #3. Update Kobo csvs: "other" values
    #3a. Funders
    new_funders = impl_activity_form_data[['program','funder_other']] 
    funders_filename = 'programs_funders'
    funders_file = list(filter(lambda x: x['metadata']['filename'] == f'{funders_filename}.csv', impl_activity_form_existing_csvs))[0]
    merged_funders_data = merge_other_choices(new_funders,funders_file,['program','name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,funders_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_funders_data,funders_filename)
    #3b. Implementers
    new_implementers = impl_activity_form_data[['program','implementer_other']] 
    implementers_filename = 'programs_implementers'
    implementers_file = list(filter(lambda x: x['metadata']['filename'] == f'{implementers_filename}.csv', impl_activity_form_existing_csvs))[0]
    merged_implementers_data = merge_other_choices(new_implementers,implementers_file,['program','name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,implementers_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_implementers_data,implementers_filename)
    #3c. Contractors
    new_contractors = impl_activity_form_data[['program','contractor_other']] 
    contractors_filename = 'programs_contractors'
    contractors_file = list(filter(lambda x: x['metadata']['filename'] == f'{contractors_filename}.csv', impl_activity_form_existing_csvs))[0]
    merged_contractors_data = merge_other_choices(new_contractors,contractors_file,['program','name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,contractors_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_contractors_data,contractors_filename)
    #3d. Landowners
    new_landowners = impl_activity_form_data[['program','landowner_other']] 
    landowners_filename = 'programs_landowners'
    landowners_file = list(filter(lambda x: x['metadata']['filename'] == f'{landowners_filename}.csv', impl_activity_form_existing_csvs))[0]
    merged_landowners_data = merge_other_choices(new_landowners,landowners_file,['program','name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,landowners_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_landowners_data,landowners_filename)

    #4. Redeploy Implementation Activity form
    redeploy_kobo_form(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN)

    #5. Create Excel Output (Implementation_Activity.xlsx)
    #impl_activity_form_data.columns = impl_activity_form_data.columns.str.removeprefix("impl_activity_")
    impl_activity_start_date = datetime.datetime.strptime(impl_activity_form_data['startdate'],'%Y-%m-%d')
    impl_activity_form_data['ID'] = impl_activity_form_data['program'] + impl_activity_form_data['nbs'] + str(impl_activity_form_data['polygon']) + str(impl_activity_start_date.month) + str(impl_activity_start_date.year)
    drop_columns = impl_activity_form_data.columns[impl_activity_form_data.columns.str.startswith('_')]
    impl_activity_form_data.drop(drop_columns, axis=1, inplace=True)
    impl_activity_form_data.to_excel("Implementation_Activity.xlsx","Implementation Activity")

    #6. Upload Excel Output to Databricks Volume (unprocessed folder)
    databricks = WorkspaceClient()
    try:
        with open('./Implementation_Activity.xlsx', 'rb') as file:
            file_bytes = file.read()
            binary_data = io.BytesIO(file_bytes)
            databricks.files.upload(f'{DATABRICKS_IMPL_ACTIVITY_UNPROCESSED_FILE_PATH}/Implementation_Activity.xlsx', binary_data, overwrite = True)
    except:
        print("unable to upload file to Databricks volume")

    #7. Run Relevant Databricks Jobs to update table data (separate codebase)
    #8. Merge new shapefiles into existing using "UnionShapefiles" methods (call manually)
    #9. Delete data in Kobo using "DeleteKoboData" methods - in seperate file (call manually)

if __name__ == '__main__':
    main()