
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if not os.path.isfile('mycreds.txt'):
    with open('mycreds.txt','w') as f:
        f.write('{"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-07-29T16:47:41Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "expires_in": 3599, "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}')

parent_id='15KEW4Oqi_5xuaVI97YMuLVhXnpmgrE3A'
gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
# if gauth.credentials is None:
#     # Authenticate if they're not there
#     gauth.LocalWebserverAuth()
if gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

# drive = GoogleDrive(gauth)

def authorize_drive():
    # global drive
    global gauth
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    # if gauth.credentials is None:
    #     # Authenticate if they're not there
    #     gauth.LocalWebserverAuth()
    if gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)

    return drive


# def validate_parent_id(parent_id):
#     global drive
#     file_list = drive.ListFile({'q': f"title='{folder_name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"}).GetList()
#         if len(file_list) > 1:
#             raise ValueError('There are multiple folders with that specified folder name')
#         elif len(file_list) == 0:
#             raise ValueError('No folders match that specified folder name')


def upload_to_drive(list_files,parent_id):
    # global drive
    drive = authorize_drive()
    # parent_id = ''# parent id
    drive_files = drive.ListFile({'q': "'%s' in parents and trashed=false"%parent_id}).GetList()
    drive_files = {f['title']:f for f in drive_files}
    for path in list_files:
        if not os.path.isfile(path): continue
        d,f = os.path.split(path)
        # check if file already exists and trash it
        if f in drive_files:
                drive_files[f].Trash()

        file = drive.CreateFile({'title': f, 'parents': [{'id': parent_id}]})
        file.SetContentFile(path)
        file.Upload()
