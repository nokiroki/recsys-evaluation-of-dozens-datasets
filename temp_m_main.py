from pathlib import Path

import hydra

from omegaconf import DictConfig

from src.preprocessing import ClassicDataset
from src.utils.processing import data_split
from src.utils.metrics.nfold_utils import get_nfold_by_method
from src.utils.logging import get_logger

logger = get_logger(name=__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    cfg_data = cfg["dataset"]
    cfg_library = cfg["library"]

    dataset = ClassicDataset()
    dataset.prepare(cfg_data)

    train_interactions = data_split(
        dataset.prepared_data, cfg_data, is_implicit=True
    )[0][0].tocsc()
    library_path = cfg_library["name"] + '/'

    if (sub_model := cfg_library["name"] + "_model") in cfg_library:
        cfg_library = cfg_library[sub_model]
        library_path += cfg_library["name"] + '/'


    common_way = Path(library_path + cfg_data["name"] + '/')

    path_load = Path("results/predicts").joinpath(common_way)
    path_save = Path("results/metrics").joinpath(common_way)

    if not path_load.exists():
        logger.error("Results do not exist, aborting...")
        return
    path_save.mkdir(parents=True, exist_ok=True)
    
    if "enable_optimization" in cfg_library:
        iters = (True, False)
    else:
        iters = (None,)

    for is_optimised in iters:
        get_nfold_by_method(
            path_load,
            path_save,
            is_optimised,
            train_interactions,
            [5, 10, 20, 100]
        )


if __name__ == "__main__":
    main()