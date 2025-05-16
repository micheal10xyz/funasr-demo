from funasr import AutoModel
import time
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need

model = AutoModel(model="iic/speech_sanm_kws_phone-xiaoyun-commands-online",
                  keywords="小云小云",
                  output_dir="./outputs/debug",
                  device='cpu',
                  chunk_size=[4, 8, 4],
                  encoder_chunk_look_back=0,
                  decoder_chunk_look_back=0,
                 )

wakeup_audio_files = [f'wakeup{i}.wav' for i in range(1, 7)]


for file_path in wakeup_audio_files:
    start = time.time()
    res = model.generate(input= './audio/' + file_path)
    elapsed = time.time() - start
    print(f'res is {res}, elapsed {elapsed}')

