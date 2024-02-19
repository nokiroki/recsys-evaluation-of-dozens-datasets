"""Loading utils module"""
from pathlib import Path
from urllib.parse import urlencode
from zipfile import ZipFile
import os

from tqdm.auto import tqdm

from omegaconf import DictConfig

import requests

# from src.utils.logging import get_logger

# logger = get_logger(name=__name__)


data_links = {
    "amazon_books": "https://disk.yandex.ru/d/rjfonRhNRTsZsw",
    "amazon_cds": "https://disk.yandex.ru/d/KYiMAaHs-l5ghQ",
    "amazon_finefoods": "https://disk.yandex.ru/d/AwlKO4SloMvIYw",
    "amazon_tv": "https://disk.yandex.ru/d/WOKJbWzEF2JLGg",
    "beeradvocate": "https://disk.yandex.ru/d/d3l4Mk-r2YpYfg",
    "brightkite": "https://disk.yandex.ru/d/724VPddbo-_ddw",
    "dianping": "https://disk.yandex.ru/d/d92ve_6H0xpsmw",
    "douban_books": "https://disk.yandex.ru/d/ofzLMLrV8Dbicw",
    "douban_movies": "https://disk.yandex.ru/d/rptqu8CiWnJ0jw",
    "douban_music": "https://disk.yandex.ru/d/ECc0iGI35LkRXg",
    "epinions": "https://disk.yandex.ru/d/WVWEuJS8JOpc6w",
    "food": "https://disk.yandex.ru/d/5U-xEIcxSVes4A",
    "foursquare": "https://disk.yandex.ru/d/bHCktkQWGZz7xg",
    "goodreads": "https://disk.yandex.ru/d/-CbGuH3W-YZuxA",
    "gowalla": "https://disk.yandex.ru/d/PaPpcpwkWcHRHg",
    "hotelrec": "https://disk.yandex.ru/d/UOQtZMqGmfyn3g",
    "kuairec_full": "https://disk.yandex.ru/d/zoy_KAn6JcNpew",
    "kuairec_small": "https://disk.yandex.ru/d/WL2yGJlCDlHkCA",
    "lfm": "https://disk.yandex.ru/d/Kt2a2nqLTjbtPA",
    "movielens_1m": "https://disk.yandex.ru/d/m1lgusnzfVtEdg",
    "movielens_10m": "https://disk.yandex.ru/d/KnijlDtKtzTLug",
    "movielens_20m": "https://disk.yandex.ru/d/tQFAqruRSRuwgA",
    "mts_library": "https://disk.yandex.ru/d/4liY8aIs6hqfhg",
    "netflix": "https://disk.yandex.ru/d/hR0RgwhQqCcS7A",
    "otto": "https://disk.yandex.ru/d/yQkDLvI9VDBnlg",
    "ratebeer": "https://disk.yandex.ru/d/Dsg7n8fzzGYmRw",
    "reddit": "https://disk.yandex.ru/d/4Ur4BPyQBKxsCA",
    "rekko": "https://disk.yandex.ru/d/MJ2a5pXp3tuDKg",
    "retail": "https://disk.yandex.ru/d/i0V2KoBvNmZJnQ",
    "tafeng": "https://disk.yandex.ru/d/H9iPgWsUAiTWbw",
    "tmall": "https://disk.yandex.ru/d/pa5ngW_Xj_cEQA",
    "twitch": "https://disk.yandex.ru/d/jz-xQrkVpCEtXQ",
    "xwines": "https://disk.yandex.ru/d/IrCMLxFWHYwGaA",
    "yahoo": "https://disk.yandex.ru/d/3CBPKH0TCPfwOA",
    "yelp": "https://disk.yandex.ru/d/vLW7QsTsprfSZA",
    "yoochoose": "https://disk.yandex.ru/d/395GLkRlzpBuRA",
    "sber_bank": "https://disk.yandex.ru/d/JCBnx9iYRTVdYQ",
    "sber_smm": "https://disk.yandex.ru/d/xUmVwOtMsur8Jg",
    "sber_zvuk": "https://disk.yandex.ru/d/ZQcuyhJRN-e02Q"
}


def load_dataset(cfg: DictConfig, download_raw: bool = False) -> None:
    """Load dataset depending on the chosen dataset from config.

    Args:
        cfg (DictConfig): OmegConf config with dataset information.
        download_raw (bool, optional): Download raw dataset. Unavailable yet. Defaults to False.

    Raises:
        ValueError: In the current version will be raised if "download_raw" option was chosen.
        NotImplementedError: No such dataset in the cloud
    """
    name: str = cfg["name"]
    data_src = Path("data")
    data_src.mkdir(exist_ok=True)

    if name not in data_links:
        raise ValueError("Not existing dataset")

    # TODO raw data
    if download_raw:
        raise NotImplementedError()

    url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?" + \
            urlencode({"public_key": data_links[name]})

    download_url = requests.get(url, timeout=10).json()["href"]

    with open("temp.zip", 'wb') as file:
        for i, chunk in enumerate(
            requests.get(download_url, stream=True, timeout=10).iter_content(1024)
        ):
            file.write(chunk)
            if i < 1024:
                output = f"{round(i, 2)} K"
            elif 1024 <= i < 1024 ** 2:
                output = f"{round(i / 1024, 2)} M"
            else:
                output = f"{round(i / 1024 ** 2, 2)} G"
            print(f"Total downloaded - " + output.rjust(9), end="\r")

    with ZipFile("temp.zip") as zfile:
        zfile.extractall(data_src)

    os.remove("temp.zip")


if __name__ == "__main__":
    for name in tqdm(data_links):
        if not Path(f"../../../data/{name}").exists():
            load_dataset({"name": name})
