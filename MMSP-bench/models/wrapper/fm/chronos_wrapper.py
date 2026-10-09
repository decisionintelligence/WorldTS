import logging

from gluonts.model import QuantileForecast
from gluonts.model.forecast_generator import SampleForecastGenerator
import torch
from chronos.chronos_bolt import ChronosBoltModelForForecasting
from gluonts.mx import DistributionOutput
from gluonts.torch.distributions import QuantileOutput
from transformers import AutoConfig
from chronos.chronos_bolt import ResidualBlock
from .base import FMWrapperBase

logger = logging.getLogger(__name__)


class ChronosWrapper(FMWrapperBase):
    @property
    def context_length(self):
        return self.model.chronos_config.context_length

    def __init__(self, model_path, ds_freq: str, prediction_length: int, *args, **kwargs):
        super().__init__()
        # self.freq = timesfm.freq_map(ds_freq)
        self._model_path = model_path
        self.load_pretrained_model(model_path, *args, **kwargs)
        self.prediction_length = prediction_length
        self.extra_length = self.prediction_length - self.model.chronos_config.prediction_length
        self.ds_freq = ds_freq
        if self.extra_length > 0:
            config = self.model.config
            self.extra_patch_embedding = ResidualBlock(
                in_dim=config.d_model,
                h_dim=config.d_ff,
                out_dim=self.model.num_quantiles * self.extra_length,
                act_fn_name=config.dense_act_fn,
                dropout_p=config.dropout_rate,
            )
        self.register_buffer("quantiles_tensor", torch.FloatTensor(self.quantiles))
        self.quantile_output = QuantileOutput

    @property
    def quantiles(self):
        return self.model.chronos_config.quantiles

    @property
    def model_dim(self):
        return self.model.model_dim

    def tokenize(self, inputs):
        context = inputs["context"]
        mask = inputs["observed_values"]
        batch_size = context.size(0)
        context, loc_scale = self.model.instance_norm(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.model.dtype)
        mask = mask.to(self.model.dtype)

        # patching
        patched_context = self.model.patch(context)
        patched_mask = torch.nan_to_num(self.model.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = (
                patched_mask.sum(dim=-1) > 0
        )  # (batch_size, patched_seq_length)

        input_embeds = self.model.input_patch_embedding(patched_context)
        if self.model.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.model.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.model.dtype),
                    torch.ones_like(reg_input_ids).to(self.model.dtype),
                ],
                dim=-1,
            )
        return {"token_embeddings": input_embeds, "attention_mask": attention_mask, "loc_scale": loc_scale}

    def encode(self, inputs):
        # model_output = self.stacked_transformer(model_input, patched_padding)
        encoder_outputs = self.model.encoder(
            attention_mask=inputs["attention_mask"],
            inputs_embeds=inputs["token_embeddings"],
        )
        hidden_states = encoder_outputs[0]
        return {"token_embeddings": inputs["token_embeddings"], "attention_mask": inputs["attention_mask"],
                "encoded_embeddings": hidden_states,
                "loc_scale": inputs["loc_scale"]}

    def predict(self, inputs):
        ctx_input = inputs["token_embeddings"]
        target_attention_mask = inputs["attention_mask"]
        hidden_states = inputs["encoded_embeddings"]
        batch_size = ctx_input.size(0)
        sequence_output = self.model.decode(ctx_input, target_attention_mask, hidden_states)
        quantile_preds_shape = (
            batch_size,
            self.model.num_quantiles,
            self.model.chronos_config.prediction_length,
        )
        quantile_preds = self.model.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )[:, :, :self.prediction_length]
        if self.extra_length > 0:
            extra_quantile_preds_shape = (
                batch_size,
                self.model.num_quantiles,
                self.extra_length,
            )
            extra_quantile_preds = self.extra_patch_embedding(sequence_output).view(
                *extra_quantile_preds_shape
            )
            quantile_preds = torch.cat([quantile_preds, extra_quantile_preds], dim=-1)
        return {"prediction": quantile_preds,
                "loc_scale": inputs["loc_scale"]}

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def loss(self, inputs):
        # normalize target
        target = inputs["future_target"]
        quantile_preds = inputs["prediction"]
        target_mask = inputs["future_observed_values"]
        loc_scale = inputs["loc_scale"]
        target, _ = self.model.instance_norm(target, loc_scale)
        target = target.unsqueeze(1)  # type: ignore
        assert self.prediction_length >= target.shape[-1]

        target = target.to(quantile_preds.device)
        target_mask = (
            target_mask.unsqueeze(1).to(quantile_preds.device)
            if target_mask is not None
            else ~torch.isnan(target)
        )
        target[~target_mask.bool()] = 0.0

        # pad target and target_mask if they are shorter than model's prediction_length
        # if self.model.chronos_config.prediction_length > target.shape[-1]:
        #     padding_shape = (
        #         *target.shape[:-1],
        #         self.model.chronos_config.prediction_length - target.shape[-1],
        #     )
        #     target = torch.cat(
        #         [target, torch.zeros(padding_shape).to(target)], dim=-1
        #     )
        #     target_mask = torch.cat(
        #         [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
        #     )

        loss = (
                2 * torch.abs((target - quantile_preds) *
                              (
                                      (target <= quantile_preds).float()
                                      - self.quantiles_tensor.view(1, len(self.quantiles), 1))
                              )
                * target_mask.float()
        )
        if self.training:
            # mask out the loss that is larger than a threshold
            loss = torch.where(loss > 3, torch.zeros_like(loss), loss)
        # loss = loss.mean(dim=-2)  # Mean over prediction horizon
        # loss = loss.sum(dim=-1)  # Sum over quantile levels
        loss_mean = loss.mean()  # Mean over batch
        return loss_mean

    def load_pretrained_model(self, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting"
            )
            class_ = ChronosBoltModelForForecasting

        self.model: ChronosBoltModelForForecasting = class_.from_pretrained(*args, **kwargs)

    def forecast_generator(self):
        return SampleForecastGenerator()

    def process_forecast(self, outputs):
        # Unscale predictions
        quantile_preds = outputs["prediction"]
        loc_scale = outputs["loc_scale"]
        quantile_preds_shape = quantile_preds.shape
        batch_size = quantile_preds_shape[0]
        quantile_preds = self.model.instance_norm.inverse(
            quantile_preds.reshape(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return quantile_preds

    # def __getstate__(self):
    #     # Return a subset of attributes that can be pickled
    #     state = self.__dict__.copy()
    #     # Remove the model attribute since it might contain unpicklable components
    #     if 'model' in state:
    #         del state['model']
    #     # Store the model path instead
    #     state['_model_path'] = getattr(self, '_model_path', None)
    #     return state
    #
    # def __setstate__(self, state):
    #     # Restore instance attributes
    #     self.__dict__.update(state)
    #     # If we have a model path, reload the model
    #     if hasattr(self, '_model_path') and self._model_path:
    #         self.load_pretrained_model(self._model_path)

    def __getnewargs_ex__(self):
        # Store constructor arguments
        return (self._model_path, getattr(self, 'ds_freq', None), self.prediction_length), {}
