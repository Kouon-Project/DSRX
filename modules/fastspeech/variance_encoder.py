import torch
import torch.nn as nn
from torch.nn import functional as F

import modules.compat as compat
from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.duration import (
    DurationDDPM,
    DurationGroupCodec,
    DurationRectifiedFlow,
    GENERATIVE_DURATION_TYPES,
    get_duration_type,
    get_duration_type_config,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, DurationPredictor
from utils.hparams import hparams
from utils.phoneme_utils import PAD_INDEX


class FastSpeech2Variance(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams['predict_dur']
        self.linguistic_mode = 'word' if hparams['predict_dur'] else 'phoneme'
        self.use_lang_id = hparams['use_lang_id']
        dur_hparams = hparams['dur_prediction_args']
        self.duration_type = get_duration_type(dur_hparams)

        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        if self.use_lang_id:
            self.lang_embed = Embedding(hparams['num_lang'] + 1, hparams['hidden_size'], padding_idx=0)

        if self.predict_dur:
            self.onset_embed = Embedding(2, hparams['hidden_size'])
            self.word_dur_embed = Linear(1, hparams['hidden_size'])
            if self.duration_type in GENERATIVE_DURATION_TYPES:
                type_hparams = get_duration_type_config(dur_hparams, self.duration_type)
                self.max_duration_group_size = int(type_hparams.get('max_group_size', 8))
                if self.max_duration_group_size < 2:
                    raise ValueError('max_group_size must be at least 2 for generative duration models.')
                self.group_position_embed = Embedding(
                    self.max_duration_group_size + 1,
                    hparams['hidden_size'],
                    padding_idx=0
                )
                self.group_size_embed = Embedding(
                    self.max_duration_group_size + 1,
                    hparams['hidden_size'],
                    padding_idx=0
                )
        else:
            self.ph_dur_embed = Linear(1, hparams['hidden_size'])

        self.encoder = FastSpeech2Encoder(
            hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'], num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'], rel_pos=hparams.get('rel_pos', False), 
            use_rope=hparams.get('use_rope', False),
            attention_type=hparams.get('enc_attention_type', 'normal')
        )

        if self.predict_dur:
            self.midi_embed = Embedding(128, hparams['hidden_size'])
            type_hparams = get_duration_type_config(dur_hparams, self.duration_type)
            if self.duration_type == 'deterministic':
                self.dur_predictor = DurationPredictor(
                    in_dims=hparams['hidden_size'],
                    n_chans=type_hparams['hidden_size'],
                    n_layers=type_hparams['num_layers'],
                    dropout_rate=type_hparams['dropout'],
                    kernel_size=type_hparams['kernel_size'],
                    offset=type_hparams['log_offset'],
                    dur_loss_type=type_hparams['loss_type'],
                    arch=type_hparams['arch']
                )
            else:
                codec = DurationGroupCodec(
                    smoothing_frames=type_hparams.get('smoothing_frames', 0.5),
                    logit_scale=type_hparams.get('logit_scale', 2.0),
                    logit_clip=type_hparams.get('logit_clip', 5.0),
                    min_frames=type_hparams.get('min_frames', 1)
                )
                backbone_type = compat.get_backbone_type(hparams, nested_config=type_hparams)
                backbone_args = compat.get_backbone_args(type_hparams, backbone_type=backbone_type)
                common_args = {
                    'codec': codec,
                    'sampling_algorithm': type_hparams.get(
                        'sampling_algorithm', 'ddim' if self.duration_type == 'ddpm' else 'rk2'
                    ),
                    'sampling_steps': type_hparams.get(
                        'sampling_steps', 20 if self.duration_type == 'ddpm' else 8
                    ),
                    'sampling_seed': type_hparams.get('sampling_seed', 1234),
                    'backbone_type': backbone_type,
                    'backbone_args': backbone_args,
                }
                if self.duration_type == 'ddpm':
                    self.dur_predictor = DurationDDPM(
                        timesteps=type_hparams.get('timesteps', 1000),
                        schedule_type=type_hparams.get('schedule_type', 'cosine'),
                        max_beta=type_hparams.get('max_beta', 0.01),
                        **common_args
                    )
                else:
                    self.dur_predictor = DurationRectifiedFlow(
                        time_scale_factor=type_hparams.get('time_scale_factor', 1000),
                        **common_args
                    )

    @staticmethod
    def _duration_group_features(ph2word):
        """Return zero-padded position and size indices for every phoneme."""
        valid = ph2word > 0
        positions = torch.zeros_like(ph2word)
        for index in range(1, ph2word.shape[1]):
            same_group = valid[:, index] & (ph2word[:, index] == ph2word[:, index - 1])
            positions[:, index] = torch.where(
                same_group,
                positions[:, index - 1] + 1,
                torch.zeros_like(positions[:, index])
            )
        max_word = int(ph2word.max().item()) if ph2word.numel() else 0
        counts = torch.zeros(
            ph2word.shape[0], max_word + 1,
            dtype=torch.long, device=ph2word.device
        ).scatter_add(1, ph2word, valid.long())
        sizes = torch.gather(counts, 1, ph2word)
        # Shift positions by one so index 0 remains reserved for padding.
        return (positions + 1) * valid, sizes * valid

    def forward(
            self, txt_tokens, midi, ph2word,
            ph_dur=None, word_dur=None,
            spk_embed=None, languages=None,
            dur_noise=None, dur_seed=None,
            dur_reference=None, dur_retake=None,
            infer=True
    ):
        """
        :param txt_tokens: (train, infer) [B, T_ph]
        :param midi: (train, infer) [B, T_ph]
        :param ph2word: (train, infer) [B, T_ph]
        :param ph_dur: (train, [infer]) [B, T_ph]
        :param word_dur: (infer) [B, T_w]
        :param spk_embed: (train) [B, T_ph, H]
        :param languages (train, infer) [B, T_ph]
        :param infer: whether inference
        :return: encoder_out, ph_dur_pred
        """
        txt_embed = self.txt_embed(txt_tokens)
        word_dur_group = word_dur
        if self.linguistic_mode == 'word':
            b = txt_tokens.shape[0]
            onset = torch.diff(ph2word, dim=1, prepend=ph2word.new_zeros(b, 1)) > 0
            onset_embed = self.onset_embed(onset.long())  # [B, T_ph, H]

            if word_dur_group is None or not infer:
                if ph_dur is None:
                    raise ValueError('ph_dur or word_dur is required for duration conditioning.')
                word_dur_group = ph_dur.new_zeros(b, ph2word.max() + 1).scatter_add(
                    1, ph2word, ph_dur
                )[:, 1:]  # [B, T_ph] => [B, T_w]
            word_dur_per_phone = torch.gather(
                F.pad(word_dur_group, [1, 0], value=0), 1, ph2word
            )  # [B, T_w] => [B, T_ph]
            if self.duration_type in GENERATIVE_DURATION_TYPES:
                word_dur_input = torch.log1p(word_dur_per_phone.float())
            else:
                word_dur_input = word_dur_per_phone.float()
            word_dur_embed = self.word_dur_embed(word_dur_input[:, :, None])
            extra_embed = onset_embed + word_dur_embed
            if self.duration_type in GENERATIVE_DURATION_TYPES:
                group_position, group_size = self._duration_group_features(ph2word)
                group_position = group_position.clamp(max=self.max_duration_group_size)
                group_size = group_size.clamp(max=self.max_duration_group_size)
                extra_embed += self.group_position_embed(group_position)
                extra_embed += self.group_size_embed(group_size)
        else:
            ph_dur_embed = self.ph_dur_embed(ph_dur.float()[:, :, None])
            extra_embed = ph_dur_embed
        if self.use_lang_id:
            lang_embed = self.lang_embed(languages)
            extra_embed += lang_embed
        encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0)

        if self.predict_dur:
            midi_embed = self.midi_embed(midi)  # => [B, T_ph, H]
            dur_cond = encoder_out + midi_embed
            if spk_embed is not None:
                dur_cond += spk_embed
            if self.duration_type == 'deterministic':
                ph_dur_pred = self.dur_predictor(
                    dur_cond, x_masks=txt_tokens == PAD_INDEX, infer=infer
                )
            else:
                ph_dur_pred = self.dur_predictor(
                    dur_cond,
                    ph2word=ph2word,
                    ph_dur=ph_dur,
                    word_dur=word_dur_group,
                    infer=infer,
                    noise=dur_noise,
                    seed=dur_seed,
                    reference_dur=dur_reference,
                    retake_mask=dur_retake
                )

            return encoder_out, ph_dur_pred
        else:
            return encoder_out, None


class MelodyEncoder(nn.Module):
    def __init__(self, enc_hparams: dict):
        super().__init__()

        def get_hparam(key, default=None):
            if key in enc_hparams:
                return enc_hparams[key]
            return hparams.get(key, default)

        # MIDI inputs
        hidden_size = get_hparam('hidden_size')
        self.note_midi_embed = Linear(1, hidden_size)
        self.note_dur_embed = Linear(1, hidden_size)

        # ornament inputs
        self.use_glide_embed = hparams['use_glide_embed']
        self.glide_embed_scale = hparams['glide_embed_scale']
        if self.use_glide_embed:
            # 0: none, 1: up, 2: down
            self.note_glide_embed = Embedding(len(hparams['glide_types']) + 1, hidden_size, padding_idx=0)

        self.encoder = FastSpeech2Encoder(
            hidden_size=hidden_size, num_layers=get_hparam('enc_layers'),
            ffn_kernel_size=get_hparam('enc_ffn_kernel_size'), ffn_act=get_hparam('ffn_act'),
            dropout=get_hparam('dropout'), num_heads=get_hparam('num_heads'),
            use_pos_embed=get_hparam('use_pos_embed'), rel_pos=get_hparam('rel_pos'),
            use_rope=get_hparam('use_rope'),
            attention_type=get_hparam('enc_attention_type', 'normal')
        )
        self.out_proj = Linear(hidden_size, hparams['hidden_size'])

    def forward(self, note_midi, note_rest, note_dur, glide=None):
        """
        :param note_midi: float32 [B, T_n], -1: padding
        :param note_rest: bool [B, T_n]
        :param note_dur: int64 [B, T_n]
        :param glide: int64 [B, T_n]
        :return: [B, T_n, H]
        """
        midi_embed = self.note_midi_embed(note_midi[:, :, None]) * ~note_rest[:, :, None]
        dur_embed = self.note_dur_embed(note_dur.float()[:, :, None])
        ornament_embed = 0
        if self.use_glide_embed:
            ornament_embed += self.note_glide_embed(glide) * self.glide_embed_scale
        encoder_out = self.encoder(
            midi_embed, dur_embed + ornament_embed,
            padding_mask=note_midi < 0
        )
        encoder_out = self.out_proj(encoder_out)
        return encoder_out
