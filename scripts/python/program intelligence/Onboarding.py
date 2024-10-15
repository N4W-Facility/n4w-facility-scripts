from boxsdk import JWTAuth, Client
import pandas as pd
from dotenv import load_dotenv, dotenv_values
from databricks.sdk import WorkspaceClient
from apiMethods.KoboInputs import fetch_kobo_data,fetch_kobo_media_files,fetch_kobo_media_content,delete_kobo_media_file,upload_kobo_media_file,redeploy_kobo_form
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

def main():
#    Databricks = WorkspaceClient()
    #created_table = w.tables.get(full_name="curated_nature_for_water_stg.reports.contracting")
    box_client = Client(box_config)
    temp_file = 'FileFromBoxTemp.xlsx'

    for item in box_client.folder('289124919729').get_items():
        if (item.name == 'Initial_Onboarding.xlsx'):
            with open(temp_file, 'wb') as open_file:
                box_client.file(item.id).download_to(open_file)
                open_file.close()

    overview_df = pd.read_excel(f'./{temp_file}',sheet_name='Program_Overview')
    partners_df = pd.read_excel(f'./{temp_file}',sheet_name='Key_Partners')
    funders_df = pd.read_excel(f'./{temp_file}',sheet_name='Key_Funding_Sources')
    kpis_df = pd.read_excel(f'./{temp_file}',sheet_name='Program_KPIs')
    nbs_df = pd.read_excel(f'./{temp_file}',sheet_name='NbS_Overview')

    #Update: programs, programs_contractors, programs_funders, programs_implementers
    ## programs_implementation_polygons, programs_partners
    impl_activity_form_existing_csvs = fetch_kobo_media_files(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/files', KOBO_TOKEN)
    programs_file = list(filter(lambda x: x['metadata']['filename'] == 'programs.csv', impl_activity_form_existing_csvs))[0]
    

    pause = 'pause'

    #2. Update Kobo csv: polygon choices for implementation activity
    #2a. Fetch existing Kobo csv files content
    #existing_polygons = fetch_kobo_media_content(existing_polygon_file['content'],KOBO_TOKEN)

if __name__ == '__main__':
    main()

