import os
import time
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
conversation_history = []

@app.route('/')
def home():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "index.html not found.", 404

def get_street_address(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    headers = {"User-Agent": "AITourGuideTest/1.0"}
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 18}
    try:
        response = requests.get(url, params=params, headers=headers).json()
        address = response.get("address", {})
        road = address.get("road", "")
        return road if road else "this immediate area"
    except Exception as e:
        return "this immediate area"

def get_prominent_wikipedia_places(lat, lon, radius=250):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": radius,
        "gslimit": 50,  
        "format": "json"
    }
    headers = {
        "User-Agent": "AITourGuideTest/1.0"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status() 
        data = response.json()
        places = data.get("query", {}).get("geosearch", [])
        return [place["title"] for place in places]
    except Exception as e:
        print(f"\n[Wikipedia API Error]: {e}")
        return []

def generate_voice_response(prompt_text, is_follow_up=False):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is missing."

    from google import genai
    from google.genai import types
    
    client = genai.Client()
    global conversation_history
    
    messages = []
    for turn in conversation_history:
        messages.append(types.Content(role="user", parts=[types.Part.from_text(text=turn['user'])]))
        messages.append(types.Content(role="model", parts=[types.Part.from_text(text=turn['model'])]))
    
    messages.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)]))
    
    system_instruction = """
    You are an expert NYC walking tour guide, acting like the Wikipedia places tab. Be historical, direct, and avoid flowery language. Aim for exactly 3-4 sentences.

    CRITICAL RULES:
    1. GEOGRAPHY: Start by stating the street the user is on.
    2. FIRST PROMPT LOGIC: If this is the first prompt in a session, give 1 sentence of high-level history about the street, then immediately point out a specific, prominent landmark from the provided data list. Give its exact address and physical appearance.
    3. SUBSEQUENT LOGIC: For following prompts, point out a brand new landmark from the list.
    4. ANTI-HALLUCINATION: You MUST trust the provided Wikipedia list. Assume they are in the user's immediate physical vicinity. Do not invent details.
    """
    
    if is_follow_up:
        system_instruction += " Answer the user's specific question about the street or building they are looking at."

    # Robust Retry & Exponential Backoff Loop
    max_retries = 3
    delay = 1 

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.4,
                )
            )
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            print(f"[Gemini API Attempt {attempt + 1} Failed]: {e}")
            
            # If it's a transient overload error or network glitch, wait and try again
            if "503" in error_msg or "unavailable" in error_msg or "resource_exhausted" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Wait 1s, then 2s
                    continue
            
            # Return a clean user-facing string if all retries are exhausted
            return "The AI servers are running slow right now. Please tap the button again in a moment."

@app.route('/api/location', methods=['POST'])
def handle_location():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    
    street_name = get_street_address(lat, lon)
    print(f"\n[Location] User is physically on: {street_name}")
    
    landmarks = get_prominent_wikipedia_places(lat, lon)
    print(f"[Wikipedia Data] Found {len(landmarks)} nearby locations.")
    
    if not landmarks:
        prompt = f"""
        The user is standing outside on {street_name} in the Upper East Side. 
        There are no prominent Wikipedia landmarks on this specific block. 
        Describe the general history of this part of the Upper East Side. Give the next closest landmark and where it is
        Give exactly 2-4 sentences.
        """
    else:
        landmark_list = ", ".join(landmarks)
        prompt = f"""
        The user is standing outside on {street_name}. 
        Here is the raw list of nearby Wikipedia locations: {landmark_list}.
        
        YOUR TASK:
        1. INTERNAL ADDRESS CHECK: Cross-reference every single item in that list with your internal database. 
        2. EXCLUSION: You MUST completely eliminate any landmark that is not located directly on {street_name}. If an item is on 78th, 80th, or 5th Ave, it is strictly forbidden.
        3. SELECTION: From the remaining items that are strictly on {street_name}, select the single most prominent one.
        4. OUTPUT: Act as their tour guide and describe that specific landmark in exactly 3-4 sentences.
        """

    script = generate_voice_response(prompt, is_follow_up=False)
    
    global conversation_history
    conversation_history.append({"user": f"[Context: On {street_name}. Near: {landmarks[:5]}]", "model": script})
    if len(conversation_history) > 5:
        conversation_history.pop(0)
        
    return jsonify({"script": script})

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.json
    user_speech = data.get('text')
    print(f"\n[User Spoke]: {user_speech}")
    
    script = generate_voice_response(user_speech, is_follow_up=True)
    
    global conversation_history
    conversation_history.append({"user": user_speech, "model": script})
    
    return jsonify({"script": script})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
