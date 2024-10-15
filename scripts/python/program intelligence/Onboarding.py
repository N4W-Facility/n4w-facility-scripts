from boxsdk import JWTAuth, Client
import pandas as pd
import numpy as np
from dotenv import load_dotenv, dotenv_values
from databricks.sdk import WorkspaceClient
from apiMethods.KoboInputs import fetch_kobo_media_files,fetch_kobo_media_content,delete_kobo_media_file,upload_kobo_media_file,redeploy_kobo_form
import pykobo
import requests
import io
import json

#Load Environment Variables for Databricks
load_dotenv()
#Box Setup
box_config = JWTAuth.from_settings_file('config.json')
#Kobo Setup
config_data = []
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

URL_KOBO = config_data['kobo_api_url']
API_VERSION = config_data['kobo_api_version']
KOBO_TOKEN = config_data['kobo_token']
USERNAME = config_data['kobo_username']
IMPLEMENTATION_ACTIVITY_FORM_UID = config_data['kobo_implementation_activity_form_uid']
DATABRICKS_IMPL_ACTIVITY_UNPROCESSED_FILE_PATH = config_data['databricks_implementation_activity_unprocessed_path']
BOX_UNPROCESSED_FOLDER_ID = config_data["box_unprocessed_folder_id"]
BOX_PROCESSED_FOLDER_ID = config_data["box_processed_folder_id"]

def merge_choices(new_data, choice_file, choice_file_fieldnames):
    existing_data = fetch_kobo_media_content(choice_file['content'],KOBO_TOKEN)
    updated_data = pd.concat([existing_data[choice_file_fieldnames], new_data], ignore_index=True)
    updated_data.drop_duplicates(inplace=True)
    updated_data.replace('', np.nan, inplace=True)
    updated_data.dropna(inplace=True)
    return updated_data

def main():
    box_client = Client(box_config)
    temp_file = 'FileFromBoxTemp.xlsx'
    box_file_to_move = {}

    for item in box_client.folder(BOX_UNPROCESSED_FOLDER_ID).get_items():
        if (item.name == 'Initial_Onboarding.xlsx'):
            box_file_to_move = box_client.file(item.id)
            with open(temp_file, 'wb') as open_file:
                box_client.file(item.id).download_to(open_file)
                open_file.close()

    overview_df = pd.read_excel(f'./{temp_file}',sheet_name='Program_Overview')
    partners_df = pd.read_excel(f'./{temp_file}',sheet_name='Key_Partners')
    funders_df = pd.read_excel(f'./{temp_file}',sheet_name='Key_Funding_Sources')
    nbs_df = pd.read_excel(f'./{temp_file}',sheet_name='NbS_Overview')
    #kpis_df = pd.read_excel(f'./{temp_file}',sheet_name='Program_KPIs')

    #Update: programs, programs_funders, programs_implementers, programs_contractors, programs_nbs
    #Ask Erik about: polygons, project managers vs partners, specific invasive species, nbs methods, 
    #Also: use "funder" or "funding stream" for choice?
    #Also: age vs gender breakdown: when to gather preference? what to use as default?
    impl_activity_form_existing_csvs = fetch_kobo_media_files(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files', KOBO_TOKEN)
    
    #Program Names
    programs_filename = 'programs'
    programs_file = list(filter(lambda x: x['metadata']['filename'] == f'{programs_filename}.csv', impl_activity_form_existing_csvs))[0]
    merged_programs_data = merge_choices(overview_df['ProgramName'],programs_file,['name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,programs_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_programs_data,programs_filename)

    #Funders
    funders_filename = 'programs_funders'
    funders_file = list(filter(lambda x: x['metadata']['filename'] == f'{funders_filename}.csv', impl_activity_form_existing_csvs))[0]
    merged_funders_data = merge_choices(funders_df[['ProgramName','FundingStream']],funders_file,['program','name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,funders_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_funders_data,funders_filename)

    #NbS
    nbs_filename = 'programs_nbs'
    nbs_file = list(filter(lambda x: x['metadata']['filename'] == f'{nbs_filename}.csv', impl_activity_form_existing_csvs))[0]
    nbs_df['NbsLowerUnderscore']=nbs_df['NbsType'].str.lower().replace(' ','_')
    nbs_df['AgeBreakdown']=0
    nbs_df['GenderBreakdown']=0
    merged_nbs_data = merge_choices(nbs_df[['ProgramName','NbsType','NbsLowerUnderscore','AgeBreakdown','GenderBreakdown']],nbs_file,['program','name','value','age_group_breakdown','gender_breakdown'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,nbs_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_nbs_data,nbs_filename)

    ### FILTER FOR IMPLEMENTERS AND CONTRACTORS!!

    #Implementers
    implementers_filename = 'programs_implementers'
    implementers_file = list(filter(lambda x: x['metadata']['filename'] == f'{implementers_filename}.csv', impl_activity_form_existing_csvs))[0]
    implementers_df = partners_df[partners_df['Type'] == 'Implementer']
    merged_implementers_data = merge_choices(implementers_df['ProgramName','Name'],implementers_file,['program','name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,implementers_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_implementers_data,implementers_filename)

    #Contractors
    contractors_filename = 'programs_contractors'
    contractors_file = list(filter(lambda x: x['metadata']['filename'] == f'{contractors_filename}.csv', impl_activity_form_existing_csvs))[0]
    contractors_df = partners_df[partners_df['Type'] == 'Contractor']
    merged_contractors_data = merge_choices(contractors_df['ProgramName','Name'],contractors_file,['program','name'])
    delete_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files',KOBO_TOKEN,contractors_file['uid'])    
    upload_kobo_media_file(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN,merged_contractors_data,contractors_filename)
    
    pause = 'pause'
    redeploy_kobo_form(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}',KOBO_TOKEN)

    #Upload File to Databricks "Unprocessed" folder
    databricks = WorkspaceClient()
    try:
        with open(f'./{temp_file}', 'rb') as file:
            file_bytes = file.read()
            binary_data = io.BytesIO(file_bytes)
            databricks.files.upload(f'{DATABRICKS_IMPL_ACTIVITY_UNPROCESSED_FILE_PATH}/Initial_Onboarding.xlsx', binary_data, overwrite = True)
    except:
        print("unable to upload file to Databricks volume")

    #Move File to "Processed" folder in Box
    destination_folder = box_client.folder(BOX_PROCESSED_FOLDER_ID)
    box_moved_file = box_file_to_move.move(parent_folder=destination_folder)
    print(f'File "{box_moved_file.name}" has been moved into folder "{box_moved_file.parent.name}"')

if __name__ == '__main__':
    main()

