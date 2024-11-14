import pandas as pd
import pykobo
import requests
import io
import json
#import chardet
#Reference forum post: https://community.kobotoolbox.org/t/how-to-replace-csv-media-file-or-delete-existing-media-file-using-kobo-v2-api/33631/13

config_data = []
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

URL_KOBO = config_data['kobo_api_url']
API_VERSION = config_data['kobo_api_version']
KOBO_TOKEN = config_data['kobo_token']
USERNAME = config_data['kobo_username']

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

def update_kobo_data(url,token,submission_ids, data):
    headers = {'Authorization': f'Token {token}'}
    params = {'format': 'json'}
    payload = {
        'submission_ids': submission_ids,
        'data': data
    }
    try:
        res = requests.patch(
            url=url,
            data={'payload': json.dumps(payload)},
            params=params,
            headers=headers
        )
    except:
        print("Unable to update record(s).")

def delete_kobo_data(url, token): 
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.delete(url, headers=headers)
        json_response = json.loads(response.text)
    except:
        print("Unable to delete kobo data. Please do so manually.")

def fetch_kobo_media_files(url, token):
 #Get existing media associated with destination form
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
    #Get existing media associated with destination form
    try:
        headers = {'Authorization': f'Token {token}'}
        response = requests.get(url, headers=headers)
        media_existing = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        media = pd.concat([media, media_existing], ignore_index=True)
    except:
        print("Unable to fetch media")
    return media

def delete_kobo_media_file(url, token, existing_file_id):
    #Delete File
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

def parse_overview_to_implementation_choices(new_choice_data, existing_choice_data):
    pause = 'pause'
    #Test parsing single field separated by semicolons
    wips_implementers = pd.DataFrame(columns=['wip_name','wip_id','wip_uuid','implementer'],data=None)
    wips_funders = pd.DataFrame(columns=['wip_name','wip_id','wip_uuid','funder'],data=None)
    wips_contractors = pd.DataFrame(columns=['wip_name','wip_id','wip_uuid','contractor'],data=None)
    wips_project_managers = pd.DataFrame(columns=['wip_name','wip_id','wip_uuid','contractor'],data=None)
    wips_nbs = pd.DataFrame(columns=['wip_name','wip_id','wip_uuid','nbs_name','nbs_id','nbs_uuid'],data=None)
 #   wips_implementation_zones = pd.DataFrame(columns=['wip_name','wip_id','wip_uuid','implementer'],data=None)

    for index, row in new_choice_data.iterrows():
        wip_name = row['wip_name']
        wip_id = row['_id']
        wip_uuid = row['_uuid']
        parsed_implementers = row['wip_implementers'].split(';')
        parsed_funders = row['wip_funders'].split(';')
        #parsed_contractors = row['wip_contractors'].split(';')
        parsed_project_managers = row['wip_projectmanagers'].split(';')
        array_nbs = row['wip_nbs_data']

        for nbs in array_nbs:
            nbs_type = nbs['wip_nbs_data/wip_nbs/wip_nbs_type']
            nbs_approach = nbs['wip_nbs_data/wip_nbs/wip_nbs_approach']

        for implementer in parsed_implementers:
            wips_implementers.loc[len(wips_implementers.index)] = {
                    'wip_name': wip_name.strip(),
                    'implementer': implementer.strip(),
                    'wip_id': wip_id,
                    'wip_uuid': wip_uuid
                }
    updated_implementers = pd.concat([existing_choice_data, wips_implementers], ignore_index=True)
    updated_implementers.drop_duplicates(inplace=True)

    pause = 'pause'

def main():
    #relevant forms
    wip_overview_form_uid = 'aykuxSFN2XYpF4Bh4khN5g'
    implementation_plan_form_uid = 'aVF9A9E2PXvqVpoCBEPMku'

    #Test values
    test_delete_file_uid = 'afQCd3utpqwCwz6dZ6mtMhC'
    test_file = 'C:\\Users\\erin.wilcox\\Downloads\\currencies.csv'

    new_data = pd.read_csv(test_file, encoding = 'Windows-1252')
    #End Test Values
    
#    input_form_data = fetch_kobo_data(URL_KOBO,KOBO_TOKEN,API_VERSION,wip_overview_form_uid)
#    output_form_media_files = fetch_kobo_media_files(f'{URL_KOBO}api/v2/assets/{implementation_plan_form_uid}/files', KOBO_TOKEN)

#    for file in output_form_media_files:
#        if 'wips_' in file['metadata']['filename']:
#            existing_choices = fetch_kobo_media_content(file['content'],KOBO_TOKEN)
#            #determine how to know which choices to parse
#            parse_overview_to_implementation_choices(input_form_data, existing_choices) #- return dataframe
#    delete_kobo_media_files(f'{URL_KOBO}api/v2/assets/{implementation_plan_form_uid}/files', KOBO_TOKEN,test_delete_file_uid)
#    upload_kobo_media_files(f'{URL_KOBO}api/v2/assets/{implementation_plan_form_uid}',KOBO_TOKEN,new_data,'currencies')
#    redeploy_kobo_form(f'{URL_KOBO}api/v2/assets/{implementation_plan_form_uid}',KOBO_TOKEN)

if __name__ == '__main__':
    main()

