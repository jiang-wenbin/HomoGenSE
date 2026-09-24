from pathlib import Path
from omegaconf import OmegaConf
import whisper
from platform import node
from utils import find_available_gpu, read_scp
import torch.utils.data as tdata
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import torch
import jiwer
from whisper.normalizers import EnglishTextNormalizer
import os

def init():
    if "CUDA_VISIBLE_DEVICES" not in os.environ: # slrum will set it
        os.environ['CUDA_VISIBLE_DEVICES'] = find_available_gpu()    
    print("Hostname: {}, CUDA_VISIBLE_DEVICES: [{}], device_count: {}".format(
        node(), os.environ.get('CUDA_VISIBLE_DEVICES', None), torch.cuda.device_count()))


class LibriSpeech(tdata.Dataset):
    def __init__(self, conf, data_dir, n_mels=80, wav='wav'):
        self.n_mels = n_mels
        wav_scp = conf.get("wav_scp", "../clean/text_tmp.txt")
        data_dir = Path(data_dir).expanduser()
        wav_dict = read_scp(data_dir.joinpath(wav_scp))
        self.dataset = [] # wav_id, wav_path, text
        for wav_id in wav_dict:
            wav_path = data_dir.joinpath(f"{wav_id}.{wav}")
            wav_path.exists() or print(f"Warning: {wav_path} not exists!")
            self.dataset.append((wav_id, wav_path, wav_dict[wav_id]))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, wav_path, text = self.dataset[idx]
        # print(wav_path)
        audio, sample_rate = sf.read(wav_path)
        audio = audio.astype(np.float32)
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        return (mel, text)
    

_MODEL_CACHE = {}


def load_model_cached(name):
    # load each whisper model only once, even when ASR() is called repeatedly
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = whisper.load_model(name)
    return _MODEL_CACHE[name]


def ASR(conf):
    init()
    model = load_model_cached(conf["model"])
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    subset_list = conf.get("subset", "-5db,0db,5db,10db,15db").split(",")
    data_root = Path(conf.get("data_root")).expanduser()
    result_path = Path(conf.get("result_path", "exp/LibriSpeech/whisper/"))
    result_path.mkdir(parents=True, exist_ok=True)
    tag = conf.get("tag", "LibriSpeech")
    for subset in subset_list:
            dataset = LibriSpeech(conf, data_root.joinpath(subset), model.dims.n_mels, 
                                  wav=conf.get("wav_ext", "wav"))
            csv_path = result_path.joinpath(f"{tag}_{subset.replace('/', '_')}.csv")
            if csv_path.exists():
                try:
                    if len(pd.read_csv(csv_path)) == len(dataset):
                        print(f"[{tag}/{subset}] skip: {csv_path} already complete")
                        continue
                except Exception as error:
                    print(f"[{tag}/{subset}] re-run: unreadable {csv_path} ({error})")
            loader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size)            
            hypotheses = []
            references = []
            for mels, texts in tqdm(loader):
                results = model.decode(mels.to("cuda"), options)
                hypotheses.extend([result.text for result in results])
                references.extend(texts)
            data = pd.DataFrame(dict(wav_id=[wav_id for wav_id, _, _ in dataset.dataset],
                                     hypothesis=hypotheses, reference=references))
            normalizer = EnglishTextNormalizer()
            data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
            data["reference_clean"] = [normalizer(text) for text in data["reference"]]
            data.to_csv(csv_path, index=False)
            wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
            print(f"[{tag}/{subset}] WER: {wer * 100:.2f} %")


def test_LibriSpeech(conf):
    data_dir = Path(conf.get("data_root")).expanduser().joinpath("-5db")
    dataset = LibriSpeech(conf, data_dir)
    num_workers = conf.get('num_workers', 4)
    dataloader = tdata.DataLoader(dataset, shuffle=True, batch_size=conf.batch_size, 
                                  num_workers=num_workers)
    item = next(iter(dataloader))
    print("item:", len(item), ", wavshape:", item[0].shape)
    # travel through the data iterator
    for _ in tqdm(dataloader):
        pass
    
    
if __name__ == '__main__':
    conf = OmegaConf.create({
         "cmd": "ASR",
         "model": "base.en",
         "batch_size": 16,
         "data_root": "./LibriSpeech/Clean",
    })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)