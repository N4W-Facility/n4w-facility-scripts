import json
import pykobo
import requests
import os
import pandas as pd
#from pyspark.sql.connect.session import SparkSession
#from pyspark.sql.connect.dataframe import DataFrame
#import pyspark.pandas as ps

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

config_data = []
with open(os.path.join(__location__, 'config.json'), 'r') as config_file:
    config_data = json.load(config_file)

URL_KOBO = config_data['kobo_api_url']
API_VERSION = config_data['kobo_api_version']
KOBO_TOKEN = config_data['kobo_token']
USERNAME = config_data['kobo_username']
IMPLEMENTATION_ACTIVITY_FORM_UID = config_data['kobo_implementation_activity_form_uid']


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
        print("unable to fetch records")
    #Returns dataframe
    return kobo_form.data

def fetch_kobo_data_without_pykobo(url, token): 
    json_response=''
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.get(url, headers=headers)
        json_response = json.loads(response.text)
    except:
        print("Unable to fetch records")
    #Returns JSON list of objects
    records = [record for record in json_response['results']]
    df_records = pd.DataFrame(records)
    #Handle Photos URL
    for i,record in df_records.iterrows():
        attachments = record._attachments
        if len(attachments)>0:
            new_polygon_file = [x for x in attachments if x['question_xpath'] == 'new_polygon_file']
            new_subpolygon_file = [x for x in attachments if x['question_xpath'] == 'new_subpolygon_file']
            photo_area = [x for x in attachments if x['question_xpath'] == 'photo_area']
            
            new_polygon_file = new_polygon_file[0]['download_url'] if len(new_polygon_file) > 0 else ''
            new_subpolygon_file = new_subpolygon_file[0]['download_url'] if len(new_subpolygon_file) > 0 else ''
            photo_area = photo_area[0]['download_url'] if len(photo_area) > 0 else ''

            df_records.at[i,'new_polygon_file'] = new_polygon_file
            df_records.at[i,'new_subpolygon_file'] = new_subpolygon_file
            df_records.at[i,'photo_area'] = photo_area
    # Handle Columns & Datatypes to avoid upload errors
    df_columns = df_records.columns
    drop_fields = [
        '_id','formhub/uuid','__version__','meta/instanceID','_xform_id_string'
        ,'_uuid','_attachments','_status','_geolocation','_tags','_notes'
        ,'_validation_status','_submitted_by','_metadata.file_path','actual_quantity_show'
        ,'contractor_show','full_polygon_show','landowner_show','latitude_show','longitude_show'
        ,'skills_required_show','units_show','team_size_show','team_age_breakdown','team_gender_breakdown'
        ,'There_is_no_sub_catc_f_this_questionnaire','This_polygon_is_part_f_this_questionnaire'
        ,'No_implementation_pl_und_for_this_polygon','_nbs_method_description'
    ]
    numeric_fields = [
        'plan_exists', 'iap_density','latitude','longitude','actual_quantity','cost'
        ,'person_days','team_size','team_youth','team_adult','team_female','team_male'
        ,'team_adultfemale','team_adultmale','team_youthfemale','team_youthmale'
    ]
    datetime_fields = [
        'start_date','end_date'
    ]
    drop_fields = [x for x in drop_fields if x in df_columns]
    numeric_fields = [x for x in numeric_fields if x in df_columns]
    datetime_fields = [x for x in datetime_fields if x in df_columns]
    
    for field in numeric_fields:
        if field not in df_records:
            df_records[field] = pd.Series(dtype='float')

    df_records.drop(columns=drop_fields,inplace=True)
    df_records.rename(columns={'_submission_time': 'submission_time'}, inplace=True)
    df_records['ID'] = df_records['program'] + df_records['polygon'] + df_records['start_date'] + df_records['end_date']
    
    df_records[numeric_fields] = df_records[numeric_fields].apply(pd.to_numeric,errors='coerce')
    df_records[datetime_fields] = df_records[datetime_fields].apply(pd.to_datetime,errors='coerce')
    pause = 'pause'

def main():
    #impl_activity_form_data = fetch_kobo_data(URL_KOBO,KOBO_TOKEN,API_VERSION,IMPLEMENTATION_ACTIVITY_FORM_UID)
    impl_activity_form_data_nopykobo = fetch_kobo_data_without_pykobo(f'{URL_KOBO}api/v2/assets/{IMPLEMENTATION_ACTIVITY_FORM_UID}/data.json', KOBO_TOKEN)
    pause = 'test'

if __name__ == '__main__':
    main()