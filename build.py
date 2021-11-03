import PyInstaller.__main__
import os
import shutil
from entry_point.main import BEST_MODEL

# constants
DIST_PATH = "./dist/"
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, BEST_MODEL)
SOUND_DIR = 'sound'


try:
    print("Build START")
    ######################### DELETE FILES UNDER DIST_PATH #########################
    if os.path.exists(DIST_PATH):
        print(f"Removing files from {DIST_PATH}...")
        for filename in os.listdir(DIST_PATH):
            file_path = os.path.join(DIST_PATH, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {repr(e)}')
        print(f"Removing files from {DIST_PATH}...DONE")
    else:
        print(f"Creating dist folder...")
        os.makedirs(DIST_PATH)
        

    ######################### COPY FILES TO DIST #########################
    print(f"Copying files to {DIST_PATH}...")
    os.makedirs(os.path.join(DIST_PATH, MODEL_DIR))
    shutil.copy(os.path.join('.', MODEL_PATH), os.path.join(DIST_PATH, MODEL_DIR))
    shutil.copytree(os.path.join(".", SOUND_DIR), os.path.join(DIST_PATH, SOUND_DIR))
    print(f"Copying files to {DIST_PATH}...DONE")
         
    ######################### BUILD APP ########################
    print(f"Building app...")
    PyInstaller.__main__.run([
        'entry_point/app.py',
        '--onedir',
        '--noconsole',
        f'--distpath={DIST_PATH}',
        #'--paths=../torch_env/lib/site-packages/cv2/'  #D:\sw_projects\ml_projects\pytorch_hackathon\torch_env\lib\site-packages\cv2\__init__.py
        '--noupx',
        '--clean'
    ])

    # There's a folder named 'app', rename it to 'bin' in --onedir mode,
    os.rename(os.path.join(f'{DIST_PATH}', 'app'), os.path.join(f'{DIST_PATH}', 'bin'))

    # clean up
    shutil.rmtree("./build")
    os.unlink("./app.spec")
    print(f"Building app...DONE")

except Exception as e:
    print(f'Failed to build. Reason: {str(e)}')
