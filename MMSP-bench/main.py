
import json
import logging
import os
import warnings

import pandas as pd
import torch
import wandb
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from tqdm import tqdm
from models.utils import get_predictor
from lightning.pytorch.loggers import WandbLogger
from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss,
)

from utils import get_args_and_settings, dataset_properties_map, get_dataset

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main():
    args, datasets = get_args_and_settings()
    # remove the duplicates
    all_datasets = sorted(datasets)

    # In[2]:

    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]
    # suppress the logging messages
    logging.getLogger('gluonts.model.forecast').setLevel(logging.ERROR)

    model_name = args.model_name
    print(f"Running model: {model_name}")

    # 支持我们使用额外的目录作为输出目录

    # HPO launchers may run several configurations concurrently.  Keep each
    # result isolated when an explicit output directory is supplied; retain
    # the released repository layout as the default.
    output_dir = args.output_dir or f"results/{model_name}"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    import sys

    config_dict = vars(args)
    config_dict["cmd"] = " ".join(sys.argv)
    wandb_logger = WandbLogger(project=f"UniCA", name=f"{model_name}", config=config_dict)
    wandb_logger.experiment

    table = wandb.Table(columns=[
        "dataset",
        "run_id",
        "config",
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain",
        "num_variates",
    ])
    with open("naive.json", "r") as f:
        naive_results = json.load(f)
    terms = ["short"]
    normalized_results = []
    for ds_name in tqdm(all_datasets):
        logger.info(f"Processing dataset: {ds_name}")
        for term in terms:
            if (
                    term == "medium" or term == "long"
            ) and ds_name not in datasets:
                continue

            dataset, ds_config, ds_key = get_dataset(args, ds_name, term)

            if dataset.target_dim != 1 and "moirai" not in model_name:
                dataset.to_univariate()
            # dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)
            wandb_logger._prefix = ds_config

            predictor = get_predictor(args, dataset, season_length, pl_logger=wandb_logger, ds_config=ds_name)
            # Measure the time taken for evaluation
            batch_size = args.batch_size
            while True:
                try:
                    torch.cuda.empty_cache()
                    res = evaluate_model(
                        predictor,
                        test_data=dataset.test_data,
                        metrics=metrics,
                        batch_size=batch_size,
                        axis=None,
                        mask_invalid_label=True,
                        allow_nan_forecast=False,
                        seasonality=season_length,
                    )
                    table.add_data(
                        ds_config,
                        wandb_logger.experiment.id,
                        # model_name,
                        json.dumps(config_dict),
                        res["MSE[mean]"][0],
                        res["MSE[0.5]"][0],
                        res["MAE[0.5]"][0],
                        res["MASE[0.5]"][0],
                        res["MAPE[0.5]"][0],
                        res["sMAPE[0.5]"][0],
                        res["MSIS"][0],
                        res["RMSE[mean]"][0],
                        res["NRMSE[mean]"][0],
                        res["ND[0.5]"][0],
                        res["mean_weighted_sum_quantile_loss"][0],
                        dataset_properties_map[ds_key]["domain"],
                        dataset_properties_map[ds_key]["num_variates"]
                    )
                    wandb.log({f"{ds_config} MAE[0.5]": res["MAE[0.5]"][0], "epoch": 1})
                    normalize = ds_config in naive_results
                    result_row = {
                        "dataset": ds_config,
                        "MAPE_norm": res["MAPE[0.5]"][0] / naive_results[ds_config]["MAPE"] if normalize else
                        res["MAPE[0.5]"][0],
                        "MSE_norm": res["MSE[0.5]"][0] / naive_results[ds_config]["MSE"] if normalize else
                        res["MSE[0.5]"][0],
                        "MAE_norm": res["MAE[0.5]"][0] / naive_results[ds_config]["MAE"] if normalize else
                        res["MAE[0.5]"][0],
                        "CRPS_norm": res["mean_weighted_sum_quantile_loss"][0] / naive_results[ds_config][
                            "mean_CRPS"] if normalize else res["mean_weighted_sum_quantile_loss"][0],
                    }
                    normalized_results.append(result_row)
                    break
                except torch.cuda.OutOfMemoryError as e:
                    print(
                        f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                    )
                    batch_size //= 2
                    if batch_size <= 0:
                        raise e

    # 保存预测结果
    if normalized_results:
        results_df = pd.DataFrame(normalized_results)
        print("\nNormalized Results:")
        print(results_df)
        csv_path = f"{output_dir}/normalized_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    wandb.summary.update({"Datasets Evaluation": table, "finish_flag": True})


if __name__ == '__main__':
    main()
