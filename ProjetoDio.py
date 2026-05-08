import sounddevice as sd # Biblioteca para gravar áudio pelo microfone
import soundfile as sf # Biblioteca para salvar o áudio gravado em arquivo
import numpy as np # Biblioteca para manipulação de arrays (Utilizada na gravação)
import whisper # Biblioteca da OpenIA para a transcrição de áudio em texto 
import pygame # Biblioteca para a reprodução dos áudios
import os # Biblioteca para manipular variáveis de ambiente do sistema
from gtts import gTTS # Biblioteca google para converter texto em fala
import google.generativeai as genai

language = "pt" # define o idioma como português

#-------------Gravação-------------
def record(sec=5, sample_rate=44100):       # Função que grava áudio, padrão de 5 segundos a 44100 Hz
    print('Ouvindo...\n')                   # Avisa o usuário que a gravação começou
    audio = sd.rec(
        int(sec * sample_rate),     # Total de amostras = segundos x taxa de amostragem
           samplerate=sample_rate,  # Taxa de amostragem (44100 Hz = qualidade CD)
           channels=1,              # Mono (1 canal)
            dtype=np.int16           # Formato dos dados de aúdio (inteiro 16 bits)
            )
    
    sd.wait()                               # Aguarda a gravação terminar completamente
    file_name = 'request_audio.wav'         # Define o nome do arquivo de saída
    sf.write(file_name, audio, sample_rate) # Salva o áudio gravado no arquivo .wav
    print(f'Áudio salvo em: {file_name}\n') # Confirma que o arquivo foi salvo 
    return file_name                        # Retorna o caminho do arquivo para uso posterior

#-------------Reprodução-------------
def reproduzir(file_name):                  # Função que reproduz um arquivo de áudio
    print(f'▶ Reproduzindo áudio...\n')     # Avisa o usuário que o áudio está sendo reproduzido
    pygame.mixer.init()                     # Inicializa o módulo de áudio do pygame
    pygame.mixer.music.load(file_name)      # Carrega o arquivo de áudio na memória 
    pygame.mixer.music.play()               # Inicia a reprodução do áudio
    while pygame.mixer.music.get_busy():    # Fica em loop enquanto o áudio estiver tocando
        pygame.time.Clock().tick(10)        # Espera 10ms a cada verificação para não sobrecarregar a CPU
    pygame.mixer.quit()                     # Libera o arquivo de áudio após a reprodução

#-------------Transcrição-------------
def transcrever(file_name):                 # Função que transcreve o áudio em texto
    print('Transcrevendo...\n')             # Avisa o usuário que começou a transcrição
    model = whisper.load_model("small")     # Carrega o modelo "small" do Whisper (Velocidade e precisão são equilibradas)
    result = model.transcribe(file_name, fp16=False, language=language)   #Transcreve o áudio (fp=False desativa a meia precisão, necessario em CPUs)
    return result["text"]                   # Retorna apenas o texto transcrito 

##-------------Gemini-------------
def perguntar_gemini(transcription):       # Função que envia o texto transcrito ao ChatGPT
    print('Consultando o Gemini...\n')
    genai.configure(api_key='API_KEY')     # Definindo a API Key que utilizaremos 
    model = genai.GenerativeModel('gemini-2.5-flash')   #Configura a versão do Gemini que é utilizada
    response = model.generate_content(transcription)    # Envia o texto transcrito para o Gemini e armazena a resposta

    return response.text  # Retorna o texto da resposta gerado pelo Gemini

#-------------Resposta em voz-------------

def falar(texto):                             # Função que converte a resposta do ChatGPT em voz
    print(f'Resposta: {texto}\n')             # Exibe a resposta no terminal
    gtts_object = gTTS(text=texto, lang=language, slow=False)   # Cria o objeto de síntese de voz com o texto e idioma definidos
    response_audio = "response_audio.mp3"     # Altera o nome do arquivo (mp3 é o unico formato suportaddo pelo gTTS)
    gtts_object.save(response_audio)          # Salva o áudio sintetizado no arquivo
    reproduzir(response_audio)                # Reproduz o áudio da resposta

#-------------Execução-------------
record_file = record()                        # Grava o áudio do usuário e salva o caminho do arquivo
reproduzir(record_file)                       # Reproduz o áudio para o usuário

transcription = transcrever(record_file)      # Transcreve o áudio gravado em texto
print(f"\nVocê disse: {transcription}\n")       # Exibe no terminal o que foii transcrito 

gemini_response = perguntar_gemini(transcription) # Envia a transcrição ao ChatGPT e obtem a resposta
falar(gemini_response)                       # Converte a resposta em voz e reproduz