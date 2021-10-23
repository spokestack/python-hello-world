import logging
import os

import requests
from spokestack.io.pyaudio import PyAudioOutput
from spokestack.profile.wakeword_asr import WakewordSpokestackASR
from spokestack.tts.clients.spokestack import TextToSpeechClient
from spokestack.tts.manager import TextToSpeechManager

from const import DEFAULT_MODEL_URL, FILE_NAMES, KEY_ID, KEY_SECRET, SAVE_PATH

_LOGGER = logging.getLogger(__name__)


def main():
    # download wakeword models
    download_models(DEFAULT_MODEL_URL)
    # initialize the speech pipeline
    pipeline = WakewordSpokestackASR.create(
        spokestack_id=KEY_ID, spokestack_secret=KEY_SECRET, model_dir="tflite"
    )
    # initialize text to speech
    manager = TextToSpeechManager(
        client=TextToSpeechClient(key_id=KEY_ID, key_secret=KEY_SECRET),
        output=PyAudioOutput(),
    )

    # add activation handler
    @pipeline.event
    def on_activate(context):
        manager.synthesize("wake word detected", "text", "demo-male")

    pipeline.run()
    
def download_models(model_url):
    """Download wake word models from URL."""
    _LOGGER.info("Downloading Wake Word Models")
    # make a local save path for the models
    os.makedirs(SAVE_PATH, exist_ok=True)

    for name in FILE_NAMES:
        # retreive the model
        req = requests.get(os.path.join(model_url, name))
        # write the model locally
        with open(os.path.join(SAVE_PATH, name), "wb") as file:
            file.write(req.content)


if __name__ == "__main__":
    main()
