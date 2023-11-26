#=============================IMPORTS=============================#
from transformers import PerceiverPreTrainedModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverMultimodalPostprocessor, 
    PerceiverMultimodalPreprocessor,
    PerceiverAudioPreprocessor,
    PerceiverImagePreprocessor,
    PerceiverOneHotPreprocessor,
    PerceiverBasicVideoAutoencodingDecoder,
    PerceiverMultimodalDecoder,
    PerceiverBasicDecoder,
    PerceiverClassificationDecoder,
    PerceiverAudioPostprocessor,
    PerceiverClassificationPostprocessor,
    PerceiverProjectionPostprocessor,
    PerceiverModel,
    PerceiverClassifierOutput,
    PERCEIVER_INPUTS_DOCSTRING,
    PERCEIVER_START_DOCSTRING,
    _CONFIG_FOR_DOC
)

from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig
#=================================================================#



@add_start_docstrings(
    """
Example use of Perceiver for multimodal (video) autoencoding, for tasks such as Kinetics-700.

[`PerceiverForMultimodalAutoencoding`] uses [`~models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor`] to
preprocess the 3 modalities: images, audio and class labels. This preprocessor uses modality-specific preprocessors to
preprocess every modality separately, after which they are concatenated. Trainable position embeddings are used to pad
each modality to the same number of channels to make concatenation along the time dimension possible. Next, one applies
the Perceiver encoder.

[`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] is used to decode the latent representation of
[`PerceiverModel`]. This decoder uses each modality-specific decoder to construct queries. The decoder queries are
created based on the inputs after preprocessing. However, autoencoding an entire video in a single forward pass is
computationally infeasible, hence one only uses parts of the decoder queries to do cross-attention with the latent
representation. This is determined by the subsampled indices for each modality, which can be provided as additional
input to the forward pass of [`PerceiverForMultimodalAutoencoding`].

[`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] also pads the decoder queries of the different
modalities to the same number of channels, in order to concatenate them along the time dimension. Next, cross-attention
is performed with the latent representation of [`PerceiverModel`].

Finally, [`~models.perceiver.modeling_perceiver.PerceiverMultiModalPostprocessor`] is used to turn this tensor into an
actual video. It first splits up the output into the different modalities, and then applies the respective
postprocessor for each modality.

Note that, by masking the classification label during evaluation (i.e. simply providing a tensor of zeros for the
"label" modality), this auto-encoding model becomes a Kinetics 700 video classifier.
""",
    PERCEIVER_START_DOCSTRING,
)
class PerceiverForMultimodalAutoencoding(PerceiverPreTrainedModel):
    def __init__(self, config: PerceiverConfig):
        super().__init__(config)

        n_audio_samples = config.num_frames * config.audio_samples_per_frame

        audio_preprocessor = PerceiverAudioPreprocessor(
            config,
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs={
                "num_bands": 192,
                "max_resolution": (n_audio_samples,),
                "sine_only": False,
                "concat_pos": True,
            },
            prep_type="patches",
            samples_per_patch=config.samples_per_patch,
        )

        image_preprocessor = PerceiverImagePreprocessor(
            config,
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs={
                "num_bands": 32,
                "max_resolution": (config.num_frames, config.image_size, config.image_size),
                "sine_only": False,
                "concat_pos": True,
            },
            prep_type="patches",
            spatial_downsample=4,
            temporal_downsample=1,
        )

        label_preprocessor = PerceiverOneHotPreprocessor(config)

        input_preprocessor = PerceiverMultimodalPreprocessor(
            min_padding_size=4,
            modalities={
                "audio": audio_preprocessor,
                "image": image_preprocessor,
                "label": label_preprocessor,
            },
            mask_probs={"image": 0.0, "audio": 0.0, "label": 1.0},
        )

        image_decoder = PerceiverBasicVideoAutoencodingDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            output_shape=config.output_shape,
            output_num_channels=config.output_num_channels,
            use_query_residual=False,
            position_encoding_only=True,
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs={
                "num_bands": 32,
                "max_resolution": (config.num_frames, config.image_size, config.image_size),
                "sine_only": False,
                "concat_pos": True,
            },
        )

        audio_decoder = PerceiverBasicDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            output_index_dims=(n_audio_samples // config.samples_per_patch,),
            output_num_channels=config.output_num_channels,
            use_query_residual=False,
            position_encoding_only=True,
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs={
                "num_bands": 192,
                "max_resolution": (n_audio_samples,),
                "sine_only": False,
                "concat_pos": True,
            },
        )

        label_decoder = PerceiverClassificationDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            use_query_residual=False,
            position_encoding_only=True,
            position_encoding_type="trainable",
            trainable_position_encoding_kwargs={
                "num_channels": config._label_trainable_num_channels,
                "index_dims": 1,
            },
        )

        decoder = PerceiverMultimodalDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            # Modality specific decoders are used ONLY to generate queries.
            # All modalties are decoded together using a unified decoder.
            modalities={
                "audio": audio_decoder,
                "image": image_decoder,
                "label": label_decoder,
            },
            num_outputs=None,
            output_num_channels=config.output_num_channels,
            use_query_residual=False,
        )

        output_postprocessor = PerceiverMultimodalPostprocessor(
            modalities={
                "audio": PerceiverAudioPostprocessor(config, in_channels=config.output_num_channels),
                "image": PerceiverProjectionPostprocessor(in_channels=config.output_num_channels, out_channels=3),
                "label": PerceiverClassificationPostprocessor(config, in_channels=config.output_num_channels),
            }
        )

        config_print = open("config_print.txt", "w")
        config_print.write(f"{config}")
        config_print.close()

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=input_preprocessor,
            decoder=decoder,
            output_postprocessor=output_postprocessor,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        subsampled_output_points: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PerceiverClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import PerceiverForMultimodalAutoencoding
        >>> import torch
        >>> import numpy as np

        >>> # create multimodal inputs
        >>> images = torch.randn((1, 16, 3, 224, 224))
        >>> audio = torch.randn((1, 30720, 1))
        >>> inputs = dict(image=images, audio=audio, label=torch.zeros((images.shape[0], 700)))

        >>> model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver")

        >>> # in the Perceiver IO paper, videos are auto-encoded in chunks
        >>> # each chunk subsamples different index dimensions of the image and audio modality decoder queries
        >>> nchunks = 128
        >>> image_chunk_size = np.prod((16, 224, 224)) // nchunks
        >>> audio_chunk_size = audio.shape[1] // model.config.samples_per_patch // nchunks
        >>> # process the first chunk
        >>> chunk_idx = 0
        >>> subsampling = {
        ...     "image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
        ...     "audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
        ...     "label": None,
        ... }

        >>> outputs = model(inputs=inputs, subsampled_output_points=subsampling)
        >>> logits = outputs.logits
        >>> list(logits["audio"].shape)
        [1, 240]

        >>> list(logits["image"].shape)
        [1, 6272, 3]

        >>> list(logits["label"].shape)
        [1, 700]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            subsampled_output_points=subsampled_output_points,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            raise NotImplementedError("Multimodal autoencoding training is not yet supported")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )