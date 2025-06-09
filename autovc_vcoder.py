import torch
from tqdm import tqdm
import librosa
from hparams import hparams
from wavenet_vocoder import builder
import soundfile as sf

def build_model():
    
    model = getattr(builder, hparams.builder)(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        weight_normalization=hparams.weight_normalization,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_scales=hparams.upsample_scales,
        freq_axis_kernel_size=hparams.freq_axis_kernel_size,
        scalar_input=True,
        legacy=hparams.legacy,
    )
    return model



def wavegen(model, c=None, tqdm=tqdm, device=None):
    """Generate waveform samples by WaveNet.
    
    """

    model.eval()
    model.make_generation_fast_()

    Tc = c.shape[0]
    upsample_factor = 256
    # Overwrite length according to feature size
    length = Tc * upsample_factor

    # B x C x T
    c = torch.FloatTensor(c.T).unsqueeze(0)

    initial_input = torch.zeros(1, 1, 1).fill_(0.0)

    # Transform data to GPU
    initial_input = initial_input.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat

def synthesize_from_mel(
    mel, 
    vocoder_model, 
    sr=16000, 
    save_path=None, 
    return_tensor=False,
    device=None
):
    """
    將 mel-spectrogram 特徵還原為 waveform（語音波形）。

    參數：
        mel (np.ndarray)        : shape = [T, 80]，G 輸出的 mel 特徵
        vocoder_model (nn.Module): 已載入參數的 WaveNet vocoder 模型
        sr (int)                : 採樣率（預設 16000）
        save_path (str or None) : 若提供檔名，儲存為 .wav 檔
        return_tensor (bool)    : 若為 True，回傳 PyTorch tensor（否則為 numpy）
        device (torch.device)   : 模型推論裝置（若未指定，預設為 vocoder 所在裝置）

    回傳：
        waveform (np.ndarray 或 Tensor): 單聲道語音波形
    """

    # 檢查輸入格式
    if not isinstance(mel, np.ndarray):
        raise TypeError("mel 必須為 numpy.ndarray 格式，shape = [T, 80]")
    
    if mel.ndim != 2 or mel.shape[1] != 80:
        raise ValueError(f"mel 的形狀錯誤，期望為 [T, 80]，但收到 {mel.shape}")
    
    # 判定 device
    if device is None:
        device = next(vocoder_model.parameters()).device

    # vocoder 推論（wavegen 會自動處理 format 和 T×C transpose）
    with torch.no_grad():
        waveform = wavegen(vocoder_model.eval(), c=mel, device=device)

    # 儲存為 .wav（若指定）
    if save_path is not None:
        sf.write(save_path, waveform, samplerate=sr)
    
    # 回傳 waveform（tensor or numpy）
    if return_tensor:
        return torch.from_numpy(waveform).float()
    else:
        return waveform