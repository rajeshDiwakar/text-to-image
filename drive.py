
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if not os.path.isfile('mycreds.txt'):
    with open('mycreds.txt','w') as f:
        f.write('{"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-07-29T16:47:41Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "expires_in": 3599, "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}')

class MyDrive(object):
    def __init__(self):
        if os.path.isfile('nodrive'):
            self.upload_to_drive = self.fake_upload_to_drive
            return
        else:
            self.upload_to_drive = self._upload_to_drive

        self.parent_id='15KEW4Oqi_5xuaVI97YMuLVhXnpmgrE3A'
        self.gauth = GoogleAuth()
        # Try to load saved client credentials
        self.gauth.LoadCredentialsFile("mycreds.txt")
        # if gauth.credentials is None:
        #     # Authenticate if they're not there
        #     gauth.LocalWebserverAuth()
        if self.gauth.access_token_expired:
            # Refresh them if expired
            self.gauth.Refresh()
        else:
            # Initialize the saved creds
            self.gauth.Authorize()
        # Save the current credentials to a file
        self.gauth.SaveCredentialsFile("mycreds.txt")
        self.cached_ids = {}
        # drive = GoogleDrive(gauth)

    def fake_authorize_drive(self):
        pass

    def fake_upload_to_drive(*args,**kwargs):
        pass

    def authorize_drive(self):
        # global drive
        gauth=self.gauth
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
    def get_parent_id(self,drive,parent_path):
        parent_path = parent_path.strip('/')
        parent_id = None
        parent_id =  self.cached_ids.get(parent_path)
        if parent_id is not None:
            return parent_id

        parent_path = parent_path.split('/')
        if len(parent_path) == 1:
            self.cached_ids[parent_path[0]] = parent_path[0]
            return parent_path[0]

        drive_folders = drive.ListFile({"q": "'%s' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"%parent_path[0]}).GetList()
        drive_folders = {f['title']:f for f in drive_folders}

        parent_id, folder_name = parent_path
        if folder_name in drive_folders:
            self.cached_ids['/'.join(parent_path)] = drive_folders[folder_name]['id']
            return drive_folders[folder_name]['id']

        parent_id, folder_name = parent_path
        file = drive.CreateFile({'title': folder_name, 'parents': [{'id': parent_id}], 'mimeType':'application/vnd.google-apps.folder'})
        # file.SetContentFile(path)
        file.Upload()
        self.cached_ids['/'.join(parent_path)] = file['id']
        return file['id']

    def _upload_to_drive(self,list_files,parent_path):
        # global drive
        drive = self.authorize_drive()
        # parent_id = ''# parent id
        parent_id = self.get_parent_id(drive,parent_path)
        drive_files = drive.ListFile({'q': "'%s' in parents and trashed=false"%parent_id}).GetList()
        drive_files = {f['title']:f for f in drive_files}
        for path in list_files:
            if not os.path.isfile(path):
                print('%s does not exist'%path)
                continue
            d,f = os.path.split(path)
            # check if file already exists and trash it
            if f in drive_files:
                    drive_files[f].Delete()
            elif f.startswith('events.out.tfevents'):
                pat = '.'.join(f.split('.')[:-2])
                for df in drive_files:
                    if df.startswith(pat):
                        drive_files[df].Delete()

            file = drive.CreateFile({'title': f, 'parents': [{'id': parent_id}]})
            file.SetContentFile(path)
            file.Upload()

if __name__=='__main__':
    import sys
    print('uploading to drive')
    mdrive = MyDrive()
    drive_parent_id='15KEW4Oqi_5xuaVI97YMuLVhXnpmgrE3A'
    mdrive.upload_to_drive([sys.argv[0]],drive_parent_id)
    print('done')
