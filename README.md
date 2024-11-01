# END-TO-END-MEDICAL-CHATBOT
# End-to-end-Medical-Chatbot-using-Llama2

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medchat python=3.8 -y
```

```bash
conda activate medchat
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
ASTRADB_API_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ASTRADB_END-POINT_API = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model from Groq cloud:
  llama models - your choice




```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2 from Groq
- AstraDB

