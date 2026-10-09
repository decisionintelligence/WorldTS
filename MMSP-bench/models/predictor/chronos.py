import numpy as np
import torch
from chronos import BaseChronosPipeline
from gluonts.itertools import batcher
from gluonts.model import SampleForecast
from tqdm import tqdm


class ChronosPredictor:
    def __init__(
            self,
            model_path,
            num_samples: int,
            prediction_length: int,
            *args,
            **kwargs,
    ):
        # print('args:', args)
        # print('kwargs:', kwargs)
        print("prediction_length:", prediction_length)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            *args,
            **kwargs,
        )
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(
            self,
            test_data_input,
            batch_size: int = 256,
            limit_prediction_length: bool = True,
    ):

        pipeline = self.pipeline
        while True:
            try:
                # Generate forecast samples
                forecast_samples = []
                for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    forecast_samples.append(
                        pipeline.predict(
                            context,
                            prediction_length=self.prediction_length,
                            # num_samples=self.num_samples,
                            # limit_prediction_length=False,  # We disable the limit on prediction length.
                        ).numpy()
                    )
                forecast_samples = np.concatenate(forecast_samples)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        # Convert forecast samples into gluonts SampleForecast objects
        sample_forecasts = []
        for item, ts in zip(forecast_samples, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return sample_forecasts
