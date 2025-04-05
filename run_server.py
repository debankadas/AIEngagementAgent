import subprocess
import sys
import os

# Ensure the script is run from the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Command to run the Uvicorn server
# - app.main:app: Specifies the module and the FastAPI app instance
# - --host 0.0.0.0: Makes the server accessible on the network
# - --port 8000: Specifies the port
# - --reload: Enables auto-reloading for development (watches for file changes)
command = [
    sys.executable,  # Use the current Python interpreter to run uvicorn
    "-m", "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8000"
    # "--reload" # Removed hot reload flag
]

print(f"Project root: {project_root}")
print(f"Running command: {' '.join(command)}")

try:
    # Execute the command
    # stdout and stderr are inherited, so Uvicorn output will appear in the console
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()  # Wait for the process to complete (e.g., if manually stopped)
except FileNotFoundError:
    print("Error: 'uvicorn' command not found.")
    print("Please ensure Uvicorn is installed in your environment:")
    print(f"  pip install uvicorn[standard]")
except KeyboardInterrupt:
    print("\nServer stopped by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Ensure the process is terminated if the script exits unexpectedly
    if 'process' in locals() and process.poll() is None:
        process.terminate()
        process.wait()
    print("Script finished.")