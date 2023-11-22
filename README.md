
## What does this project do?

- In Microsoft word document, it will automatically sense bi-lingual texts(bangla, english) and speak out Bengali & English text, by leveraging the TTS model from locally.


## Run Locally

- Clone the project

```bash
  git clone https://github.com/masum035/Facebook-Meta-AI-TTS-Integration-in-dotnet-framework
```

- Go to the project directory

```bash
  cd Meta-AI-TTS-Integration\Python-TTS-Integration
```
- Create a virtual environment
```bash
pip install virtualenv
python -m venv venv
```

- Activate that virtual Environment
```bash
  cd venv/Scripts
  activate
  cd ..
```

- Install dependencies

```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
```
- In case, if you want to deactivate the virtual environment
```bash
  cd venv/Scripts
  deactivate
  cd ..
```

- Download Facebook-Meta-AI PyTorch model for Bangla:

```bash
  https://dl.fbaipublicfiles.com/mms/tts/ben.tar.gz
```

This will download a zip file, extract it in the project folder's (Python-TTS-Integration) directory, after extracting the file, you will see `ben` folder

## Running Tests

To run tests, see the `Main_Converter.py`



## Acknowledgements

- [Facebook-Meta-AI-Blog](https://ai.meta.com/blog/multilingual-model-speech-recognition/)


