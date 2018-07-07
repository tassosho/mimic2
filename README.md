# mimic2

This is a fork of [keithito/tacotron](https://github.com/keithito/tacotron)
with changes specific to Mimic 2 applied.


## Background

Google published a paper, [Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/pdf/1703.10135.pdf),
where they present a neural text-to-speech model that learns to synthesize speech directly from
(text, audio) pairs. However, they didn't release their source code or training data. This is an
attempt to provide an open-source implementation of the model described in their paper.

The quality isn't as good as Google's demo yet, but hopefully it will get there someday :-).
Pull requests are welcome!


## Quick Start

### Installing dependencies

#### using docker (recommended)
1. make sure you have docker installed

1. Build Docker
   
   the Dockerfile comes with a gpu option or cpu option. If you want to use the GPU in docker make sure you have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed

   gpu: `docker build -t mycroft/mimic2:gpu -f gpu.Dockerfile .`
   
   cpu: `docker build -t mycroft/mimic2:gpu -f cpu.Dockerfile .`

2. Run Docker

   gpu: `nvidia-docker run -it -p 3000:3000 mycroft/mimic2:gpu`
   
   cpu: `docker run -it -p 3000:3000 mycroft/mimic2:cpu`

#### manually
1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 or 1.4.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```


### Using a pre-trained model
   **NOTE this model will only work if you switch out the LocationSensitiveAttention layer for the BahdanauAttention layer in tacotron.py

1. **Download and unpack a model**:
   ```
   curl http://data.keithito.com/data/speech/tacotron-20170720.tar.bz2 | tar xjC /tmp
   ```

2. **Run the demo server**:
   ```
   python3 demo_server.py --checkpoint /tmp/tacotron-20170720/model.ckpt
   ```

3. **Point your browser at localhost:3000**
   * Type what you want to synthesize



### Training

*Note: you need at least 40GB of free disk space to train a model.*

1. **Download a speech dataset.**

   The following are supported out of the box:
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
    * [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)
    * [M-ailabs](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/)

   You can use other datasets if you convert them to the right format. See [TRAINING_DATA.md](TRAINING_DATA.md) for more info.


2. **Unpack the dataset into `~/tacotron`**

   After unpacking, your tree should look like this for LJ Speech:
   ```
   tacotron
     |- LJSpeech-1.1
         |- metadata.csv
         |- wavs
   ```

   or like this for Blizzard 2012:
   ```
   tacotron
     |- Blizzard2012
         |- ATrampAbroad
         |   |- sentence_index.txt
         |   |- lab
         |   |- wav
         |- TheManThatCorruptedHadleyburg
             |- sentence_index.txt
             |- lab
             |- wav
   ```
   
   For M-AILABS follow the directory structure from [here](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/)

3. **Preprocess the data**
   ```
   python3 preprocess.py --dataset ljspeech
   ```
     * other datasets can be used i.e. `--dataset blizzard` for Blizzard data
     * for the mailabs dataset, do `preprocess.py --help` for options. Also note that mailabs uses sample_size of 16000

4. **Train a model**
   ```
   python3 train.py
   ```

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command
   line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"`.
   Hyperparameters should generally be set to the same values at both training and eval time.


5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`.

6. **Synthesize from a checkpoint**
   ```
   python3 demo_server.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   Replace "185000" with the checkpoint number that you want to use, then open a browser
   to `localhost:9000` and type what you want to speak. Alternately, you can
   run [eval.py](eval.py) at the command line:
   ```
   python3 eval.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   If you set the `--hparams` flag when training, set the same value here.

