import pandas as pd
from dotenv import load_dotenv, dotenv_values
from databricks.connect import DatabricksSession
from boxsdk import JWTAuth, Client
from office365.sharepoint.client_context import ClientCredential, ClientContext, UserCredential
import json
#from office365.sharepoint.files.file import File
from databricks.sdk import WorkspaceClient

#Load Environment Variables
load_dotenv()

#Box Setup
#box_config = JWTAuth.from_settings_file('config.json')

# SharePoint Setup
#config_data = []
#with open('config.json', 'r') as config_file:
#    config_data = json.load(config_file)

#sp_username = config_data['sp_username']
#sp_password = config_data['sp_password']
#sp_url = config_data['sp_url']
#sp_user_credentials = UserCredential(sp_username,sp_password)

def main():
#    file_name = 'FileTemp.xlsx'

#    spark = DatabricksSession.builder.getOrCreate()
#    spark_df = spark.read.table("samples.nyctaxi.trips")
#    df = spark_df.toPandas()
#    df.to_excel("FileTemp.xlsx")

#    box_client = Client(box_config)
#    sp_ctx = ClientContext(sp_url).with_credentials(sp_user_credentials)

#    with open(file_name, 'rb') as content_file:
#        file_content = content_file.read()
#        target_folder = sp_ctx.web.get_folder_by_server_relative_url(f'{sp_url}/Shared Documents/Reporting')
#        target_folder.upload_file(file_name, file_content).execute_query()

#    folder_id = '9046d30e-8257-49e7-b64a-ae6339ae98eb'
#    with open(file_name, 'rb') as open_file:
#        box_client.folder(folder_id).upload('test_report.xlsx')
#        open_file.close()

    #0x0120000B41966A7A91A54A802AC5FC9CFB0DC2

    #Demo Databricks SDK - may be easier to get a file this way
    w = WorkspaceClient()
    #created_table = w.tables.get(full_name="curated_nature_for_water_stg.reports.contracting")
    pause = 'pause'

if __name__ == '__main__':
    main()






