import sys
import os

print("===================================================")
echo_green = lambda text: print(f"\033[92m{text}\033[0m")
echo_red = lambda text: print(f"\033[91m{text}\033[0m")

print("Checking Python Environment...")
print(f"Python version: {sys.version}")
print(f"Execution path: {sys.executable}")
print("---------------------------------------------------")

dependencies = [
    ("torch", lambda: f"PyTorch version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})"),
    ("cv2", lambda: f"OpenCV version: {cv2.__version__}"),
    ("mediapipe", lambda: "MediaPipe imported successfully!"),
    ("deepface", lambda: f"DeepFace version: {deepface.__version__}"),
    ("fastapi", lambda: f"FastAPI imported successfully!"),
    ("streamlit", lambda: f"Streamlit version: {streamlit.__version__}"),
]

success = True
for lib_name, print_info in dependencies:
    try:
        mod = __import__(lib_name)
        globals()[lib_name] = mod
        echo_green(f"[OK] {lib_name.capitalize()} is installed.")
        print(f"     -> {print_info()}")
    except ImportError as e:
        echo_red(f"[FAIL] {lib_name.capitalize()} could not be imported.")
        print(f"       Reason: {e}")
        success = False

print("---------------------------------------------------")
if success:
    echo_green("CONGRATULATIONS: All core dependencies are verified and active!")
else:
    echo_red("WARNING: Some dependencies failed verification. Please review errors above.")
print("===================================================")
