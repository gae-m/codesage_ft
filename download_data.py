import argparse
import logging
import os
import shutil
import gdown


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_primevul(data_url: str, dst_dir: str) -> None:
    if os.path.exists(dst_dir):
        raise ValueError(f"Destination directory {dst_dir} already exists. Please remove it before downloading.")
    
    gdown.download_folder(url=data_url, output=dst_dir) # type: ignore


def download_bigvul(data_url: str, dst_dir: str) -> None:
    if os.path.exists(dst_dir):
        raise ValueError(f"Destination directory {dst_dir} already exists. Please remove it before downloading.")
    
    os.makedirs(dst_dir, exist_ok=True)
    downloaded_file =gdown.download(url=data_url, output=dst_dir) # type: ignore
    assert isinstance(downloaded_file, str), f"Expected downloaded_file to be a string, but got {type(downloaded_file)}"
    shutil.unpack_archive(downloaded_file, dst_dir)
    os.remove(downloaded_file)


ORIGIN_DIR = os.path.join(os.getcwd(), 'data', 'origin')

ORIGIN = {
    'primevul': {
        'data_url' : r'https://drive.google.com/drive/folders/19iLaNDS0z99N8kB_jBRTmDLehwZBolMY',
        'dst_dir' : os.path.join(ORIGIN_DIR, 'primevul'),
        'download_func' : download_primevul
    },
    'bigvul': {
        'data_url' : r'https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?safe=active',
        'dst_dir' : os.path.join(ORIGIN_DIR, 'bigvul'),
        'download_func' : download_bigvul
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for GVDM.")
    parser.add_argument('datasets', nargs='+', choices=['primevul', 'bigvul'], help="Datasets to download.")

    args = parser.parse_args()

    opt = ORIGIN

    for dataset in args.datasets:
        if dataset not in opt.keys():
            logger.error(f"Dataset {dataset} is not available for download.")
            continue

        data_url = opt[dataset]['data_url']
        dst_dir = opt[dataset]['dst_dir']
        download_func = opt[dataset]['download_func']

        logger.info(f"Downloading {dataset} to {dst_dir}...")
        download_func(data_url, dst_dir)
        logger.info(f"Downloaded {dataset} successfully.")


if __name__ == '__main__':
    main()
