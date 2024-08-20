import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import azure.cognitiveservices.speech as speechsdk
IMAGE_PATH = "YOUR_IMAGE_PATH"
encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

endpoint = os.getenv("ENDPOINT_URL", "https://psu-ai-service.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "psu-gpt-4o-mini")
search_endpoint = os.getenv("SEARCH_ENDPOINT", "https://psu01-search.search.windows.net")
search_key = os.getenv("SEARCH_KEY", "put your Azure AI Search admin key here")
search_index = os.getenv("SEARCH_INDEX_NAME", "psu01-index4")

# setup speech configuration 
# SPEECH_API_KEY is they key of the speech resource
speech_config = speechsdk.SpeechConfig(
  subscription=os.getenv("SPEECH_API_KEY"), 
  region="westus2"
)

# Get the text from the microphone
audio_config = speechsdk.audio.AudioConfig(
  use_default_microphone=True)
speech_config.speech_recognition_language="en-US"
speech_recognizer = speechsdk.SpeechRecognizer(
  speech_config, 
  audio_config)

print("Say something...")
speech_result = speech_recognizer.recognize_once_async().get()


token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default")
      
client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2024-05-01-preview",
)
      
completion = client.chat.completions.create(
    model=deployment,
    messages= [{"role": "user", "content": speech_result.text}],
    max_tokens=800,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False,
    extra_body={
      "data_sources": [{
          "type": "azure_search",
          "parameters": {
            "endpoint": f"{search_endpoint}",
            "index_name": "psu01-index4",
            "semantic_configuration": "default",
            "query_type": "semantic",
            "fields_mapping": {},
            "in_scope": True,
            "role_information": "User:\nHello!! I am Supriya from U.S.Bank. How may I help you today?\nsystem: \nHello Supriya! I'm interested in learning about the services offered by U.S. Bank that can help reduce expenditures for check printing. Could you provide some details on that?\nUser: \nSure. Before we go deep into the conversation, I would like to understand the company's total dollar amount of non-payroll payments for the past 12 months\nsystem: \nFor our company, XYZ Healthcare, the total dollar amount of non-payroll payments for the past 12 months is approximately $1.2 million. This includes various operational expenses, vendor payments, and service fees.\nCould you now share how U.S. Bank can assist in optimizing these expenditures, particularly regarding check printing?\nUser: \nalso, may I understand how does check initiation happens today? file upload or manual entry?\nsystem:\nCurrently, at XYZ Healthcare, our check initiation process involves a combination of manual entry and file uploads. Specifically, we handle about 60% of our check payments through manual entry, where team members input payment details directly into our accounting system. The remaining 40% is managed via file uploads from our accounts payable software, which helps streamline some of the payment processes.\nCould you explain how U.S. Bank's AP automation services can enhance this process and reduce costs associated with check printing?\n ",
            "filter": None,
            "strictness": 3,
            "top_n_documents": 5,
            "authentication": {
              "type": "api_key",
              "key": f"{search_key}"
            }
          }
        }]
    }
)
print(completion.to_json())

# Play the result on the computer's speaker
speech_config.speech_synthesis_voice_name = "en-US-AvaNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config)
speech_synthesizer.speak_text(
  completion.choices[0].message.content)
