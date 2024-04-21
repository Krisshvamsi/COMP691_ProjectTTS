#!/usr/bin/env/python3


import os
import sys
import torch
import pickle
import warnings
import torch
import torch as nn
import torch.nn.functional as F
import torchaudio
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.inference.text import GraphemeToPhoneme
from speechbrain.inference.vocoders import HIFIGAN


logger = logging.getLogger(__name__)


# Brain class for TTS training
class TTS(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def save_scale_values(self):
        """
            Saves the scale values of the encoder and decoder modules to pickle files.
            These scale values are used for tracking the scaling factors applied during
            the training process, which can be useful for analysis and debugging purposes."""
        def append_to_file(file_path, data):
            """
                    Appends data to a file using pickle serialization.

                    Parameters:
                        file_path (str): The path to the file where data will be appended.
                        data (any): The data to be appended to the file.

            """

            # Write the updated data back to the file
            with open(file_path, 'ab') as file:
                pickle.dump(data, file)

        # Paths for the encoder and decoder scale values
        encoder_file_path = os.path.join(self.hparams.save_folder, 'encoder_scale_values.pkl')
        decoder_file_path = os.path.join(self.hparams.save_folder, 'decoder_scale_values.pkl')

        # Append scale values to their respective files
        append_to_file(encoder_file_path, self.encoder_scale_values)
        append_to_file(decoder_file_path, self.decoder_scale_values)

        # Reset the lists of new values after saving
        self.new_encoder_scale_values = []
        self.new_decoder_scale_values = []

    def remember_samples(self, mel_output, mel_length):
        # Store predictions; adapt based on your actual output format
        self.remembered_samples.append((mel_output, mel_length))

    def should_remember(self, epoch):
        # Determines if samples should be remembered based on the current epoch and log interval
        return (self.hparams.progress_samples and
                epoch % self.hparams.progress_samples_interval == 0 and
                epoch >= self.hparams.progress_samples_min_run)

    def log_remembered_samples(self,stage, epoch):
        # Iterate through remembered samples to save them
        vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                       savedir="pretrained_models/hifi-gan-ljspeech")

        for i, (mel_output, mel_length) in enumerate(self.remembered_samples):

            waveform = vocoder.decode_batch(mel_output.to(self.device), mel_length,
                                            hop_len=self.hparams.hop_length )

            filename = f"{stage}_epoch{epoch}_{i}.wav"
            filepath = os.path.join(self.hparams.progress_sample_logger.output_path, filename)
            torchaudio.save(filepath, waveform.squeeze(1), self.hparams.sample_rate)

        # Optionally, clear remembered samples to save memory
        self.remembered_samples.clear()

    @torch.no_grad()
    def inference(self, encoder_emb, initial_input, max_generation_length=1000):
        """
            Performs inference to generate mel spectrograms using the trained model.

            This function generates mel spectrograms by iteratively decoding the input mel spectrogram
            and stopping when a predefined stop condition is met or the maximum generation length is reached.

            Parameters:
                encoder_emb (torch.Tensor): The encoded input phoneme sequence.
                initial_input (torch.Tensor): The initial input mel spectrogram for decoding.
                max_generation_length (int): The maximum length of the generated mel spectrogram sequence. Defaults to 1000.

            Returns:
                stop_tokens_logits (torch.Tensor): Logits indicating the stop tokens generated during inference.
                results (torch.Tensor): The concatenated tensor containing the generated mel spectrogram sequence. """

        stop_generated = False
        decoder_input = initial_input
        stop_tokens_logits = []
        sequence_length = 0

        result = []
        result.append(decoder_input)

        src_mask = torch.zeros(encoder_emb.size(1), encoder_emb.size(1), device=self.device)
        src_key_padding_mask = self.hparams.padding_mask(encoder_emb, self.hparams.blank_index)


        while not stop_generated and sequence_length < max_generation_length:
            encoded_mel = self.modules.dec_pre_net(decoder_input)
            pos_emb_dec = self.modules.pos_emb_dec(encoded_mel)
            decoder_emb = encoded_mel + pos_emb_dec

            decoder_output = self.modules.Seq2SeqTransformer(
                encoder_emb, decoder_emb, src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask)

            mel_output = self.modules.mel_lin(decoder_output)

            stop_token_logit = self.modules.stop_lin(decoder_output).squeeze(-1)

            post_mel_outputs = self.modules.postnet(mel_output.to(self.device))
            refined_mel_output = mel_output + post_mel_outputs.to(self.device)
            refined_mel_output = refined_mel_output.transpose(1, 2)

            stop_tokens_logits.append(stop_token_logit)
            stop_token_probs = torch.sigmoid(stop_token_logit)

            if torch.any(stop_token_probs[:, -1] >= self.hparams.stop_threshold):
                stop_generated = True

            decoder_input = refined_mel_output
            result.append(decoder_input)
            sequence_length += 1

        results = torch.cat(result, dim=2)
        stop_tokens_logits = torch.cat(stop_tokens_logits, dim=1)

        return stop_tokens_logits, results

    def compute_forward(self, batch, stage):
        """Runs the forward pass of the model to compute predictions.

        Arguments
        ---------
        batch : PaddedBatch
            The batch object containing input tensors.
        stage : sb.Stage
            The stage of training: sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tuple[torch.tensor, torch.tensor]
            Predicted stop token logits and refined mel spectrogram outputs.

        At validation/test time, it returns the predicted tokens as well.
            """

        id, phoneme_seq_padded, mel_specs_padded, mel_lengths, stop_token_targets_padded = batch

        encoded_phoneme = self.modules.encoder_emb(phoneme_seq_padded)
        encoder_emb = self.modules.enc_pre_net(encoded_phoneme)
        encoder_emb = encoder_emb + self.modules.pos_emb_enc(encoded_phoneme)

        if stage == sb.Stage.TEST:
            initial_input = torch.zeros(mel_specs_padded.shape[0], 80, 1, device=self.device)
            stop_tokens_logits, refined_mel_outputs = self.inference( encoder_emb, initial_input)

            vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                           savedir="pretrained_models/hifi-gan-ljspeech")
            seq_len = refined_mel_outputs.shape[2]  # The common length of all sequences if they are uniform.
            mel_lengths = torch.full((refined_mel_outputs.shape[0],), seq_len, dtype=torch.long, device=refined_mel_outputs.device)
            limited_ids = id[:self.hparams.test_batch_sample_size]
            limited_mel_outputs = refined_mel_outputs[:self.hparams.test_batch_sample_size]
            limited_mel_lengths = mel_lengths[:self.hparams.test_batch_sample_size]

            # Generate waveforms from mel spectrograms
            generated_waveforms = vocoder.decode_batch(limited_mel_outputs.to(self.device),
                                                      limited_mel_lengths, hop_len=self.hparams.hop_length)

            base_path = os.path.join(self.hparams.output_folder, "generated_waveforms")

            # Assuming hop_length and sample_rate are defined in your hparams
            hop_length = self.hparams.hop_length
            sample_rate = self.hparams.sample_rate

            for i, (waveform, mel_length) in enumerate(zip(generated_waveforms, limited_mel_lengths)):
                # Calculate the original audio length
                original_length = mel_length * hop_length

                # Truncate the waveform to the original length
                waveform = waveform[:, :original_length]

                os.makedirs(base_path, exist_ok=True)
                file_path = os.path.join(base_path, f"generated_waveform_{i}.wav")
                torchaudio.save(file_path, waveform.squeeze(1), sample_rate)

            logger.info(f"Saved samples per batch from the test set.")

        else:

            decoder_input = torch.cat([torch.zeros_like(mel_specs_padded[:, :, :1]), mel_specs_padded[:, :, :-1]],dim=2)
            encoded_mel = self.modules.dec_pre_net(decoder_input)
            pos_emb_dec = self.modules.pos_emb_dec(encoded_mel)
            decoder_emb = encoded_mel + pos_emb_dec

            # Log the scale values
            self.encoder_scale_values.append(self.modules.pos_emb_enc.scale.item())
            self.decoder_scale_values.append(self.modules.pos_emb_dec.scale.item())

            # Masking
            tgt_mask = self.hparams.lookahead_mask(decoder_emb)
            src_mask = torch.zeros(encoder_emb.shape[1], encoder_emb.shape[1], device=self.device)
            src_key_padding_mask = self.hparams.padding_mask(encoder_emb, self.hparams.blank_index)
            tgt_key_padding_mask = self.hparams.padding_mask(decoder_emb, self.hparams.blank_index)

            # Transformer
            decoder_outputs = self.modules.Seq2SeqTransformer(encoder_emb, decoder_emb, src_mask=src_mask,
                                                              tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                                              tgt_key_padding_mask=tgt_key_padding_mask)

            # Post-net computations
            mel_outputs = self.modules.mel_lin(decoder_outputs)
            stop_tokens_logits = self.modules.stop_lin(decoder_outputs).squeeze(-1)
            post_mel_outputs = self.modules.postnet(mel_outputs.to(self.device))
            refined_mel_outputs = mel_outputs + post_mel_outputs.to(self.device)
            refined_mel_outputs = refined_mel_outputs.transpose(1,2)

            if stage == sb.Stage.VALID and self.should_remember(self.hparams.epoch_counter.current):
                self.remember_samples(refined_mel_outputs, mel_lengths)

        return (stop_tokens_logits, refined_mel_outputs)


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch

            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Unpack the predictions
        stop_tokens_logits, refined_mel_outputs = predictions
        id, phoneme_seq_padded, mel_specs_padded, mel_lengths, stop_token_targets_padded = batch

        stop_loss = self.hparams.stop_loss(stop_tokens_logits, stop_token_targets_padded, mel_lengths, pos_weight=torch.tensor([self.hparams.pos_weight], device=self.device))
        mel_loss = self.hparams.mel_loss(refined_mel_outputs.float(), mel_specs_padded.float(), mel_lengths)

        #Option1
        loss = stop_loss + mel_loss

        #Option2
        # loss = (
        #         self.hparams.stop_weight * stop_loss
        #         + (1 - self.hparams.stop_weight) * mel_loss
        # )

        self.last_loss_stats[stage] = {
            "total_loss": loss.item(),
            "stop_loss": stop_loss.item(),
            "mel_loss": mel_loss.item()
        }

        if stage != sb.Stage.TRAIN:
            self.mel_error_metrics.append(id, refined_mel_outputs.float(), mel_specs_padded.float())
            self.stop_error_metrics.append(id, stop_tokens_logits, stop_token_targets_padded)

        return loss

    def on_fit_batch_start(self, batch, should_step):
        """Gets called at the beginning of each fit_batch."""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch=None):
        """Called at the beginning of each epoch, sets up metrics"""
        if stage != sb.Stage.TRAIN:
            self.mel_error_metrics = self.hparams.mel_error_stats()
            self.stop_error_metrics = self.hparams.stop_error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Performs actions at the end of each stage of training/validation/testing.

            Arguments
            ---------
            stage : sb.Stage
                The stage of training: sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
            stage_loss : float
                The loss value computed for the stage.
            epoch : int or None, optional
                The current epoch number. Defaults to None.

            At the end of training stage, it saves the scale values and logs training statistics.
            At the end of validation stage, it logs validation statistics, saves predicted samples if required,
            and performs model checkpointing based on the `mel_error`.
            At the end of testing stage, it logs testing statistics.
            """
        stage_stats = self.last_loss_stats[stage]
        current_epoch = self.hparams.epoch_counter.current

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self.save_scale_values()

        else:
            stage_stats["mel_error"] = self.mel_error_metrics.summarize("average")
            stage_stats["stop_error"] = self.stop_error_metrics.summarize("average")

        if stage == sb.Stage.VALID:
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            # Log training and validation statistics
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "optimizer": optimizer},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save predicted samples if required
            if self.should_remember(epoch):
                logger.info("Saving predicted samples")
                self.log_remembered_samples(stage, epoch)
                self.remembered_samples = []

            # Model checkpointing based on mel_error
            self.checkpointer.save_and_keep_only(meta=stage_stats, min_keys=["mel_error"])

        elif stage == sb.Stage.TEST:
            # Log testing statistics
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0

        self.encoder_scale_values = []
        self.decoder_scale_values = []

        self.last_loss_stats = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(self.hparams.progress_sample_logger.output_path, exist_ok=True)
        self.remembered_samples = []

        return super().on_fit_start()

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    """


    @sb.utils.data_pipeline.takes("phoneme_seq")
    @sb.utils.data_pipeline.provides("phoneme_seq_encoded_lst", "phoneme_seq_encoded")
    def text_pipeline(phoneme_seq):
        """Processes the transcriptions to generate proper labels"""

        phoneme_seq_encoded_lst = label_encoder.encode_sequence(phoneme_seq)
        yield phoneme_seq_encoded_lst
        phoneme_seq_encoded = torch.LongTensor(phoneme_seq_encoded_lst)
        yield phoneme_seq_encoded

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("mel_spec", "mel_lengths", "stop_token_targets")
    def mel_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig, fs = torchaudio.load(wav)
        sig = sig.squeeze(0)
        mel_spec, _ = hparams['mel_spec_feats'](audio=sig)
        # Initialize BOS and EOS tokens as additional "time steps" with the chosen values
        #bos_token = torch.ones(80, 1)
        #eos_token = torch.full((80, 1), 2)

        # Concatenate BOS token, actual mel spectrogram, and EOS token
        #bos_mel_spectrogram = torch.cat([bos_token, mel_spec], dim=1)
        #eos_mel_spectrogram = torch.cat([mel_spec, eos_token], dim=1)

        # Update the mel length to include BOS and EOS tokens
        #mel_lengths = bos_mel_spectrogram.shape[1]

        # Calculate lengths and stop tokens
        mel_lengths = mel_spec.shape[1]  # [n_mels, time] -> time
        stop_token_targets = torch.zeros(mel_lengths)
        stop_token_targets[-1:] = 1  # The last frame should have the stop token
        return mel_spec, mel_lengths, stop_token_targets

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    # The label encoder will assign a different integer to each element
    # in the output vocabulary
    label_encoder = sb.dataio.encoder.TextEncoder()
    label_encoder.update_from_iterable(hparams["lexicon"], sequence_input=False)
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[text_pipeline, mel_pipeline],
            output_keys=[
                "id",
                "phoneme_seq_encoded_lst",
                "phoneme_seq_encoded",

                "mel_spec",
                "mel_lengths",
                "stop_token_targets",

            ],
        )

        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = True

    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="phoneme_seq",
        special_labels=special_labels,
        sequence_input=True,
    )

    return datasets, label_encoder


if __name__ == "__main__":
    # Reading command line arguments

    warnings.filterwarnings("ignore")
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # We can now directly create the datasets for training, valid, and test
    datasets, label_encoder = dataio_prepare(hparams)

    # Trainer initialization
    TTS_brain = TTS(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Making label encoder accessible (needed for computer the character error rate)
    TTS_brain.label_encoder = label_encoder

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the pointpointer, training can be
    # stopped at any point, and will be resumed on next call.
    TTS_brain.fit(
        TTS_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # After fit() completes, save the scale values
    #TTS_brain.save_scale_values()

    # Load best checkpoint for evaluation
    test_stats = TTS_brain.evaluate(
        test_set=datasets["test"],
        min_key="combined loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

