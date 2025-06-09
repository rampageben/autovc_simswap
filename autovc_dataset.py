import os
import torch
import numpy as np
import librosa
from torch.utils import data


class DynamicUtterancesDataset(data.Dataset):
    def __init__(self, root_dir, len_crop=256, sr=16000, encoder=None):
        self.root_dir = root_dir
        self.len_crop = len_crop
        self.sr = sr
        self.hop_length = 256
        self.n_mels = 80
        self.encoder = encoder
        self.encoder.eval()

        # 掃描所有 speaker 資料夾與 wav 檔
        self.speakers = []
        for speaker in sorted(os.listdir(root_dir)):
            speaker_dir = os.path.join(root_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            wav_paths = [os.path.join(speaker_dir, f)
                         for f in os.listdir(speaker_dir)
                         if f.endswith(".wav")]
            if len(wav_paths) > 0:
                self.speakers.append((speaker, wav_paths))

        

    def wav_to_mel(self, path):
        wav, _ = librosa.load(path, sr=self.sr)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=self.sr, n_fft=1024, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=80, fmax=7600)
        mel_db = librosa.power_to_db(mel).T  # shape: [T, 80]
        return mel_db

    def __getitem__(self, index):
        speaker_id, wav_list = self.speakers[index]
        wav_path = np.random.choice(wav_list)
        mel = self.wav_to_mel(wav_path)

        if mel.shape[0] < self.len_crop * 2:
            raise ValueError(f"{wav_path} 太短，不能裁兩段")

        # 隨機取 content 段
        left1 = np.random.randint(0, mel.shape[0] - self.len_crop)
        mel_content = mel[left1:left1 + self.len_crop]

        # 隨機取 speaker embedding 段
        left2 = np.random.randint(0, mel.shape[0] - self.len_crop)
        mel_emb = mel[left2:left2 + self.len_crop]
        mel_emb_tensor = torch.from_numpy(mel_emb[np.newaxis, :, :]).float()  
        emb = self.encoder(mel_emb_tensor.to(next(self.encoder.parameters()).device)).detach().squeeze().cpu().numpy()


        return mel_content, emb

    def __len__(self):
        return len(self.speakers)


def get_dynamic_loader(root_dir, batch_size=16, len_crop=128, num_workers=0, encoder=None):
    dataset = DynamicUtterancesDataset(root_dir, len_crop,encoder=encoder)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
