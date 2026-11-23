
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Clé API introuvable.")

client = genai.Client(api_key=api_key)

print("Envoi de la requête à Gemini...")

try:
    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input="Réponds simplement : Bonjour, Gemini fonctionne !"
    )

    print("Réponse de Gemini :")
    print(response.output_text)

except Exception as e:
    print("Une erreur est survenue :")
    print(type(e).__name__, str(e))
