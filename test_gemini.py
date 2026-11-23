
import os
from dotenv import load_dotenv
from google import genai

# Charger les variables du fichier .env
load_dotenv()

# Récupérer la clé API
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("La clé API Gemini est introuvable.")

# Créer le client Gemini
client = genai.Client(api_key=api_key)

# Envoyer une première demande à Gemini
response = client.interactions.create(
    model="gemini-3.6-flash",
    input="Présente-toi en une phrase en français."
)

# Afficher la réponse
print("Réponse de Gemini :")
print(response.output_text)
