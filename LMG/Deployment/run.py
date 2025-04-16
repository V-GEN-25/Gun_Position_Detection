from pyngrok import ngrok
import streamlit as st
import os

# Set your Ngrok auth token here or as environment variable
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTHTOKEN")

if __name__ == "__main__":
    # Start Streamlit in a separate thread
    from threading import Thread
    import subprocess
    
    def start_streamlit():
        subprocess.run(["streamlit", "run", "app.py"])
    
    Thread(target=start_streamlit, daemon=True).start()
    
    # Set up Ngrok tunnel
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(8501).public_url
    print(f"Public URL: {public_url}")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")