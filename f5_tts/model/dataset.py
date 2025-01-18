import os
from pathlib import Path
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, Sampler

from einops import rearrange

from f5_tts.model.modules import MelSpec

# Dynamic Batch Sampler

class DynamicBatchSampler(Sampler):
    def __init__(self, data_source, max_batch_tokens, collate_fn, shuffle=True):
        self.data_source = data_source
        self.max_batch_tokens = max_batch_tokens
        self.collate_fn = collate_fn
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        
        # Shuffle indices if required
        if self.shuffle:
            random.shuffle(indices)
        
        batch = []
        cum_length = 0
        for idx in indices:
            item = self.data_source[idx]
            item_length = item['mel_spec'].shape[-1]  # or use len(item['text']) for text length

            if cum_length + item_length > self.max_batch_tokens and batch:
                yield batch
                batch = []
                cum_length = 0
            
            batch.append(idx)
            cum_length += item_length

        if batch:
            yield batch

    def __len__(self):
        return len(self.data_source)

class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            # mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["mp3"]["array"]
        sample_rate = row["mp3"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["mp3"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["mp3"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
        text = row["json"]["text"]
        # duration = row["json"]["duration"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )
    
class TextAudioDataset(Dataset):
    def __init__(
        self,
        folder,
        audio_extensions = ["wav"],
        target_sample_rate = 24_000,
        min_duration = 0.3,
        max_duration = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        self.audio_extensions = audio_extensions

        files = []
        for audio_extension in audio_extensions:
            files.extend(list(path.glob(f'**/*.{audio_extension}')))
        assert len(files) > 0, 'no files found'
        
        valid_files = []
        
        if max_duration is None:
            valid_files = files
        else:
            for file in tqdm(files):
                try:
                    text_file = Path(file).with_suffix('.normalized.txt')
                    if not os.path.exists(text_file):
                        # print(f"Missing normalized text at path: {text_file}")
                        continue
                    
                    duration = self.calculate_wav_duration(file)
                    
                    if duration > max_duration or duration < min_duration:
                        # print(f"Skipping due to duration out of bound: {duration}")
                        continue
                    else:
                        valid_files.append(file)
                except Exception as e:
                    print(e)
                    continue
        
            files = valid_files
        
        print(f"Using {len(files)} files.")
        
        self.files = files
        self.target_sample_rate = target_sample_rate
        self.mel_spectrogram = MelSpec(
            target_sample_rate = 24_000,
            filter_length = 1024,
            hop_length = 256,
            win_length = 1024,
            n_mel_channels = 100
        )
    
    def calculate_wav_duration(self, file_path):
        # assumptions
        sample_rate = 24_000
        bit_depth = 16
        num_channels = 1
        
        bytes_per_sample = bit_depth // 8
        bytes_per_second = sample_rate * num_channels * bytes_per_sample
        
        file_size = os.path.getsize(file_path)
        duration_seconds = file_size / bytes_per_second
        
        return duration_seconds

    def get_melspec(self, file):
        mel_file = file.with_suffix('.mel')
        
        if not mel_file.exists():
            audio_tensor, sample_rate = torchaudio.load(file)
            
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                audio_tensor = resampler(audio_tensor)
            
            mel_spec = self.mel_spectrogram(audio_tensor)
            mel_spec = rearrange(mel_spec, '1 d t -> d t')
            
            torch.save(mel_spec, mel_file)
            
        return torch.load(mel_file, weights_only = True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        return dict(
            mel_spec = self.get_melspec(file),
            text = file.with_suffix('.normalized.txt').read_text(encoding="utf-8").strip(),
        )


# collation

def collate_fn(batch):
    mel_specs = [item['mel_spec'].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()
    
    # round to nearest multiple of 128
    mult = 128
    max_mel_length = ((max_mel_length + mult - 1) // mult) * mult

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value = 0)
        padded_mel_specs.append(padded_spec)
    
    mel_specs = torch.stack(padded_mel_specs)

    text = [item['text'] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel = mel_specs,
        mel_lengths = mel_lengths,
        text = text,
        text_lengths = text_lengths,
    )

