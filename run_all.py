import os
import subprocess
import sys
from threading import Thread

def run_streamlit():
    os.system("streamlit run app.py")

def run_api():
    os.system("python modules/api/app.py")

if __name__ == "__main__":
    # Create two threads
    streamlit_thread = Thread(target=run_streamlit)
    api_thread = Thread(target=run_api)
    
    # Start both threads
    print("Starting Streamlit app and API server...")
    streamlit_thread.start()
    api_thread.start()
    
    try:
        # Wait for both threads to complete (they won't normally)
        streamlit_thread.join()
        api_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down both servers...")
        sys.exit(0)