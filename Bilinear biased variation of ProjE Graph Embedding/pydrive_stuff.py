from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import google.colab
from oauth2client.client import GoogleCredentials
import glob, os

#
folder_id = "1KQid9Hjp_e76dFeQaqlOON472xeqjDmC"  # backup and restore everything
mainfolder_id, datafolder_id, codesfolder_id = "1KQid9Hjp_e76dFqQaolOON472xeyjDmC", "1WJpGq8b5SkTmC2mkqSI7fS_t8d4xZggp", "1AiP_J_QM4wmbJC7mFqsgoRpllS3UV-yH"
dir_to_backup = "/root/content/projb"
mainfolder_local, datafolder_local, codesfolder_local = "/root/content/projb", "/root/content/projb/data", "/root/content/projb/codes"
mycreds_file_contents = ''
mycreds_file = 'mycreds.json'

with open(mycreds_file, 'w') as f:
    f.write(mycreds_file_contents)


def authenticate_pydrive():
    gauth = GoogleAuth()

    # https://stackoverflow.com/a/24542604/5096199
    # Try to load saved client credentials
    gauth.LoadCredentialsFile(mycreds_file)
    if gauth.credentials is None:
        # Authenticate if they're not there
        google.colab.auth.authenticate_user()
        gauth.credentials = GoogleCredentials.get_application_default()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile(mycreds_file)

    drive = GoogleDrive(gauth)
    return drive


def backup_pydrive():
    drive = authenticate_pydrive()
    ##
    ##
    ## mainfolder:
    folder_id = mainfolder_id
    dir_to_backup = mainfolder_local
    paths = list(glob.iglob(os.path.join(dir_to_backup, '**'), recursive=True))
    files2del = [itm.split('/')[-1] for itm in paths if
                 '.' in itm.split('/')[-1]]  ##warning: none-typed files won't be regarded
    # Delete existing files
    drive = authenticate_pydrive()
    files = drive.ListFile({'q': "'%s' in parents" % folder_id}).GetList()
    for file in files:
        drive = authenticate_pydrive()
        if file['title'] in files2del:
            file.Delete()
            # print("deleted file:%s"%(file['title']))
        else:
            pass  # print('thisonefailed:it seems file is actually a folder or is none-typed one!:%s'%(file['title']))
    #
    for path in paths:
        if os.path.isdir(path) or os.stat(path).st_size == 0:
            continue
        drive = authenticate_pydrive()
        file = drive.CreateFile({'title': path.split('/')[-1], 'parents':
            [{"kind": "drive#fileLink", "id": folder_id}]})
        file.SetContentFile(path)
        file.Upload()
    print('==>Backed up mainfolder.')
    ##
    ##
    ## datafolder:
    folder_id = datafolder_id
    dir_to_backup = datafolder_local
    drive = authenticate_pydrive()
    paths = list(glob.iglob(os.path.join(dir_to_backup, '**'), recursive=True))
    files2del = [itm.split('/')[-1] for itm in paths if
                 '.' in itm.split('/')[-1]]  ##warning: none-typed files won't be regarded
    # Delete existing files
    drive = authenticate_pydrive()
    files = drive.ListFile({'q': "'%s' in parents" % folder_id}).GetList()
    for file in files:
        # print('processing file %s'%(file['title']))
        drive = authenticate_pydrive()
        if file['title'] in files2del:
            file.Delete()
            # print("deleted file:%s"%(file['title']))
        else:
            pass  # print('thisonefailed:it seems file is actually a folder or is none-typed one!:%s'%(file['title']))
    for path in paths:
        if os.path.isdir(path) or os.stat(path).st_size == 0:
            continue
        drive = authenticate_pydrive()
        file = drive.CreateFile({'title': path.split('/')[-1], 'parents':
            [{"kind": "drive#fileLink", "id": folder_id}]})
        file.SetContentFile(path)
        file.Upload()
    print('=>Backed up datafolder')
    ##
    ##
    ## codesfolder:
    folder_id = codesfolder_id
    dir_to_backup = codesfolder_local
    drive = authenticate_pydrive()
    paths = list(glob.iglob(os.path.join(dir_to_backup, '**'), recursive=True))
    files2del = [itm.split('/')[-1] for itm in paths if
                 '.' in itm.split('/')[-1]]  ##warning: none-typed files won't be regarded
    # Delete existing files
    drive = authenticate_pydrive()
    files = drive.ListFile({'q': "'%s' in parents" % folder_id}).GetList()
    for file in files:
        drive = authenticate_pydrive()
        if file['title'] in files2del:
            file.Delete()
            # print("deleted file:%s"%(file['title']))
        else:
            pass  # print('thisonefailed:it seems file is actually a folder or is none-typed one!:%s'%(file['title']))
    for path in paths:
        if os.path.isdir(path) or os.stat(path).st_size == 0:
            continue
        drive = authenticate_pydrive()
        file = drive.CreateFile({'title': path.split('/')[-1], 'parents':
            [{"kind": "drive#fileLink", "id": folder_id}]})
        file.SetContentFile(path)
        file.Upload()
        print('=>Backed up codesfolder')


def save(idd='1-6g_vg1BU8jyEKiYctIs499Sp1m6d596', pat='vids'):
    ##
    ##
    folder_id = idd
    dir_to_backup = pat
    paths = list(glob.iglob(os.path.join(dir_to_backup, '**'), recursive=True))
    files2del = [itm.split('/')[-1] for itm in paths if
                 '.' in itm.split('/')[-1]]  ##warning: none-typed files won't be regarded
    # Delete existing files
    drive = authenticate_pydrive()
    #
    for path in paths:
        if os.path.isdir(path) or os.stat(path).st_size == 0:
            continue
        drive = authenticate_pydrive()
        file = drive.CreateFile({'title': path.split('/')[-1], 'parents':
            [{"kind": "drive#fileLink", "id": folder_id}]})
        file.SetContentFile(path)
        file.Upload()
    print('==>Backed up mainfolder.')


def restore_pydrive():
    drive = authenticate_pydrive()
    ##
    ##
    ## mainfolder:
    folder_id = mainfolder_id
    dir_to_backup = mainfolder_local
    local_download_path = os.path.expanduser(dir_to_backup)
    try:
        os.makedirs(local_download_path)
    except:
        pass
    file_list = drive.ListFile({'q': "'%s' in parents" % folder_id}).GetList()
    #
    for f in file_list:
        drive = authenticate_pydrive()
        # 3. Create & download by id.
        # print('title: %s, id: %s' % (f['title'], f['id']))
        isdir = '.' not in f['title']
        if isdir is False:
            fname = os.path.join(local_download_path, f['title'])
            # print('downloading to {}'.format(fname))
            f_ = drive.CreateFile({'id': f['id']})
            f_.GetContentFile(fname)
            # print('Restored %s' % f['title'])
        else:
            try:
                os.makedirs(os.path.join(local_download_path, f['title']), exist_ok=True)
            except:
                pass
    print('Restored mainfolder')
    ##
    ##
    ## datafolder:
    folder_id = datafolder_id
    dir_to_backup = datafolder_local
    local_download_path = os.path.expanduser(dir_to_backup)
    try:
        os.makedirs(local_download_path)
    except:
        pass
    file_list = drive.ListFile({'q': "'%s' in parents" % folder_id}).GetList()
    #
    for f in file_list:
        drive = authenticate_pydrive()
        # 3. Create & download by id.
        # print('title: %s, id: %s' % (f['title'], f['id']))
        isdir = '.' not in f['title']
        if isdir is False:
            fname = os.path.join(local_download_path, f['title'])
            # print('downloading to {}'.format(fname))
            f_ = drive.CreateFile({'id': f['id']})
            f_.GetContentFile(fname)
            # print('Restored %s' % f['title'])
        else:
            try:
                os.makedirs(os.path.join(local_download_path, f['title']), exist_ok=True)
            except:
                pass
    print('Restored datafolder')
    ##
    ##
    ## codesfolder:
    folder_id = codesfolder_id
    dir_to_backup = codesfolder_local
    local_download_path = os.path.expanduser(dir_to_backup)
    try:
        os.makedirs(local_download_path)
    except:
        pass
    file_list = drive.ListFile({'q': "'%s' in parents" % folder_id}).GetList()
    #
    for f in file_list:
        drive = authenticate_pydrive()
        # 3. Create & download by id.
        # print('title: %s, id: %s' % (f['title'], f['id']))
        isdir = '.' not in f['title']
        if isdir is False:
            fname = os.path.join(local_download_path, f['title'])
            # print('downloading to {}'.format(fname))
            f_ = drive.CreateFile({'id': f['id']})
            f_.GetContentFile(fname)
            # print('Restored %s' % f['title'])
        else:
            try:
                os.makedirs(os.path.join(local_download_path, f['title']), exist_ok=True)
            except:
                pass
    print('Restored codesfolder')


# print('*** authening your drive ***')
# authenticate_pydrive()
