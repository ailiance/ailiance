"""Augment chat-fr training set with short greetings + small talk.

The original chat-fr.jsonl contains long French essays / Q&A. The MiniLM
encoder learnt that "chat-fr" = "long French text", so a short prompt like
"Bonjour" is mis-classified into a random class with high confidence
(observed: music-audio at 0.99).

This script appends ~200 short greetings + small talk to chat-fr.jsonl
in the messages format the existing data uses, so re-running build_router_data.py
naturally picks them up.

Run on studio:
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/augment_short_greetings.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CLASSIFIED = Path.home() / "KIKI-Mac_tunner/data/micro-kiki/classified"
SEED = 17

# Short greetings + small talk in FR/EN/mixed. The LLM will produce a generic
# friendly reply — content of the assistant message is just there because the
# existing dataset uses messages format. Only the user content matters for
# the router training.
SHORT_GREETINGS = [
    "Bonjour",
    "Bonjour !",
    "Bonjour, comment vas-tu ?",
    "Bonjour, comment allez-vous ?",
    "Bonjour, comment ça va ?",
    "Salut",
    "Salut !",
    "Salut, ça va ?",
    "Salut, comment ça va ?",
    "Coucou",
    "Coucou, ça va ?",
    "Hello",
    "Hello!",
    "Hi",
    "Hi there",
    "Hey",
    "Hey, how are you?",
    "How are you?",
    "How are you doing?",
    "Comment allez-vous ?",
    "Comment ça va ?",
    "Ça va ?",
    "Ça va bien ?",
    "Tout va bien ?",
    "Comment se passe ta journée ?",
    "Bonsoir",
    "Bonsoir, comment vas-tu ?",
    "Bonne soirée",
    "Bonne nuit",
    "Bonne journée",
    "Bon matin",
    "Good morning",
    "Good evening",
    "Good night",
    "Good afternoon",
    "Merci",
    "Merci beaucoup",
    "Merci, c'est gentil",
    "Thanks",
    "Thank you",
    "Thank you so much",
    "Au revoir",
    "À bientôt",
    "À demain",
    "Bye",
    "Goodbye",
    "See you later",
    "See you tomorrow",
    "Quoi de neuf ?",
    "What's up?",
    "What's new?",
    "Comment puis-je t'appeler ?",
    "Tu peux te présenter ?",
    "Présente-toi",
    "Qui es-tu ?",
    "What's your name?",
    "Tell me about yourself",
    "Comment tu t'appelles ?",
    "On peut discuter ?",
    "Tu es là ?",
    "Are you there?",
    "Are you online?",
    "Tu m'entends ?",
    "Tu peux m'aider ?",
    "Peux-tu m'aider ?",
    "Can you help me?",
    "Help",
    "À l'aide",
    "Help me please",
    "Pouvez-vous m'aider s'il vous plaît ?",
    "S'il te plaît",
    "S'il vous plaît",
    "Please",
    "Yes",
    "No",
    "Oui",
    "Non",
    "Peut-être",
    "Maybe",
    "I'm not sure",
    "Je ne sais pas",
    "I don't know",
    "OK",
    "D'accord",
    "Très bien",
    "Parfait",
    "Perfect",
    "Cool",
    "Génial",
    "Super",
    "Top",
    "Excellent",
    "Désolé",
    "Désolée",
    "Pardon",
    "Sorry",
    "I'm sorry",
    "Excuse me",
    "Excusez-moi",
    "Mon ami, ça fait longtemps !",
    "Tiens, salut, ça fait plaisir de te voir",
    "Hey, ça roule ?",
    "Yo, comment va ?",
    "Salut, j'ai une question rapide",
    "Hi, I have a quick question",
    "Bonjour, j'ai besoin d'aide",
    "Hello, I need some help",
    "Tu es disponible ?",
    "Available now?",
    "On peut parler ?",
    "Tu peux discuter là ?",
    "Discutons un peu",
    "Let's chat",
    "Tu fais quoi ?",
    "What are you up to?",
    "Comment se passe ta semaine ?",
    "How's your week going?",
    "C'est sympa",
    "C'est génial",
    "That's nice",
    "That's awesome",
    "Très intéressant",
    "Interesting",
    "C'est intéressant",
    "Tell me more",
    "Dis-m'en plus",
    "Vraiment ?",
    "Really?",
    "Tu plaisantes ?",
    "Are you kidding?",
    "Sérieux ?",
    "Seriously?",
    "Wow",
    "Oh là là",
    "C'est dingue",
    "That's crazy",
    "C'est incroyable",
    "Amazing",
    "Bravo",
    "Well done",
    "Congrats",
    "Félicitations",
    "Bonne chance",
    "Good luck",
    "Take care",
    "Prends soin de toi",
    "Repose-toi bien",
    "Sleep well",
    "Bonne chance pour la suite",
    "Have a nice day",
    "Passe une bonne journée",
    "Have a good evening",
    "Bonne soirée à toi aussi",
    "Bisous",
    "Cheers",
    "À tout à l'heure",
    "À tout de suite",
    "Catch you later",
    "Bonjour, quelle heure est-il ?",
    "Quelle est la date d'aujourd'hui ?",
    "What time is it?",
    "What's today's date?",
    "Quel temps fait-il ?",
    "How's the weather?",
    "Tu connais la météo ?",
    "Bonjour, je peux te poser une question ?",
    "Hello, can I ask a question?",
    "Hi, may I ask you something?",
    "Salut, j'aimerais savoir un truc",
    "Bonjour, j'aurais une question",
    "Quel est ton modèle ?",
    "What model are you?",
    "Tu es quelle version ?",
    "Which version are you?",
    "Bonjour, dis-moi qui tu es",
    "Hi, who are you?",
    "Présente ton fonctionnement",
    "How do you work?",
    "Comment tu fonctionnes ?",
    "Tu peux faire quoi ?",
    "What can you do?",
    "Tes capacités ?",
    "Your capabilities?",
    "Liste de tes fonctions",
    "List your features",
    "Bonjour, peux-tu m'aider en français ?",
    "Hi, can we speak in French?",
    "Tu parles français ?",
    "Do you speak French?",
    "Tu parles plusieurs langues ?",
    "Multilingual?",
    "Quelles langues tu parles ?",
    "Which languages do you speak?",
]


def main() -> None:
    random.seed(SEED)
    target = CLASSIFIED / "chat-fr.jsonl"
    if not target.exists():
        print(f"ERROR: {target} not found")
        return
    before = target.read_text().count("\n")
    with target.open("a", encoding="utf-8") as f:
        for prompt in SHORT_GREETINGS:
            obj = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": (
                            "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
                            if any(c in prompt.lower() for c in ["bonjour", "salut", "coucou", "hello", "hi", "hey"])
                            else "D'accord, je suis là pour vous aider."
                        ),
                    },
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    after = target.read_text().count("\n")
    print(f"chat-fr.jsonl : {before} -> {after} lines (+{len(SHORT_GREETINGS)} short greetings)")


if __name__ == "__main__":
    main()
