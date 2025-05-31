import gdown
import time

DATABASE_URL = "https://drive.google.com/drive/folders/1OIuuI4WU_ItPLyhPavqJLRiM5rJULk9G"
DATABASE_OLD_URL ="https://drive.google.com/drive/folders/1OQOx7nq1QEqzXeL0jJuFb5JuPsFlqy2R"
MODELS_URL = "https://drive.google.com/drive/folders/1ihXUzXNqdE69XWfOXtYK-rn2SRJFOqe8"

def download_folder(url,output_folder,quiet=True):
    gdown.download_folder(url=url, output=output_folder,  quiet=quiet)        

def main(database_old=False):
    t1 = time.time()
    if database_old:
        download_folder(DATABASE_OLD_URL,"database",quiet=False)
    else:
        download_folder(DATABASE_URL,"database",quiet=False)
    t2 = time.time()
    download_folder(MODELS_URL,"models_zip",quiet=False)
    t3 = time.time()

    print(f"Download database Time taken: {t2 - t1} seconds")
    print(f"Download models Time taken: {t3 - t2} seconds")
    print(f"Total time taken: {t3 - t1} seconds")

if __name__ == '__main__':
    main()
