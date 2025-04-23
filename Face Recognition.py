import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk # For Progressbar
import cv2
import os
import face_recognition
import numpy as np
from PIL import Image, ImageTk
import time
import pickle # For saving/loading encodings
import threading
import queue

# --- Constants ---
DATASET_PATH = "dataset"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
ENCODINGS_PATH = "encodings.pkl" # File to save/load encodings
CAPTURE_COUNT = 30
RECOGNITION_THRESHOLD = 0.6

# --- Global Variables ---
known_face_encodings = []
known_face_names = []
face_cascade = None
cap = None
# Threading and Queue for Recognition
recognition_thread = None
recognition_stop_event = threading.Event()
frame_queue = queue.Queue()
# GUI State Flags
add_person_running = False
recognition_active = False # New flag to track if recognition *should* be running
# Tkinter Widgets (initialized in create_gui)
window = None
video_label = None
status_label = None
progress_bar = None
btn_start_rec = None
btn_stop_rec = None

# --- Core Functions ---

def load_face_cascade():
    """Loads the Haar Cascade classifier."""
    global face_cascade
    if not os.path.exists(CASCADE_PATH):
        messagebox.showerror("Error", f"Haar Cascade file not found at: {CASCADE_PATH}")
        return False
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        messagebox.showerror("Error", "Failed to load Haar Cascade classifier.")
        return False
    print("Haar Cascade loaded successfully.")
    return True

def ensure_dataset_dir():
    """Ensures the dataset directory exists."""
    if not os.path.exists(DATASET_PATH):
        try:
            os.makedirs(DATASET_PATH)
            print(f"Created dataset directory: {DATASET_PATH}")
        except OSError as e:
            messagebox.showerror("Error", f"Failed to create dataset directory: {e}")
            return False
    return True

def save_trained_model():
    """Saves the trained encodings and names to a file."""
    if not known_face_names:
        print("No data to save.")
        return False
    try:
        with open(ENCODINGS_PATH, 'wb') as f:
            pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)
        update_status(f"Trained model saved to {ENCODINGS_PATH}")
        print(f"Trained model saved successfully to {ENCODINGS_PATH}.")
        return True
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save encodings: {e}")
        print(f"Error saving encodings: {e}")
        return False

def load_trained_model():
    """Loads trained encodings and names from a file if it exists."""
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data.get('encodings', [])
                known_face_names = data.get('names', [])
                if known_face_encodings and known_face_names:
                    num_persons = len(set(known_face_names))
                    update_status(f"Loaded trained model for {num_persons} persons from {ENCODINGS_PATH}")
                    print(f"Loaded trained model successfully from {ENCODINGS_PATH}. Persons: {num_persons}")
                    return True
                else:
                    update_status("Encoding file is empty or corrupt. Please train the model.")
                    print("Encoding file found but empty or corrupt.")
                    return False
        except Exception as e:
            update_status(f"Error loading encodings: {e}. Please train the model.")
            messagebox.showwarning("Load Error", f"Could not load encoding file: {e}\nPlease train the model.")
            print(f"Error loading encodings: {e}")
            # Optionally delete the corrupt file
            # try: os.remove(ENCODINGS_PATH) except OSError: pass
            return False
    else:
        update_status("No pre-trained model found. Please train the model.")
        print(f"Encoding file not found at {ENCODINGS_PATH}")
        return False

def train_model():
    """Trains the face recognition model and saves it."""
    global known_face_encodings, known_face_names, progress_bar
    update_status("Starting training...")
    progress_bar['value'] = 0
    window.update_idletasks()

    known_face_encodings = []
    known_face_names = []
    image_paths = []
    person_names_for_paths = []

    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        update_status("Dataset folder not found or empty. Add persons first.")
        messagebox.showwarning("Training", "Dataset folder not found or empty. Please add persons first.")
        progress_bar['value'] = 0
        return

    # --- Collect all image paths first for progress calculation ---
    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, filename)
                    image_paths.append(image_path)
                    person_names_for_paths.append(person_name)

    if not image_paths:
        update_status("No images found in dataset folder.")
        messagebox.showwarning("Training", "No valid images found in the dataset folder.")
        progress_bar['value'] = 0
        return

    progress_bar['maximum'] = len(image_paths)
    update_status(f"Training model on {len(image_paths)} images...")
    print(f"Starting training on {len(image_paths)} images...")
    # --- ---

    images_processed = 0
    encodings_found = 0
    unique_persons = set()

    for image_path, person_name in zip(image_paths, person_names_for_paths):
        try:
            print(f"Processing {image_path} for {person_name}...")
            image = face_recognition.load_image_file(image_path)
            # Use HOG model (faster) by default. CNN is more accurate but much slower.
            face_locations = face_recognition.face_locations(image, model="hog") # or "cnn"

            if face_locations:
                # Encode the *first* face found in the image
                # Assumes dataset images ideally have one clear face
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)
                encodings_found += 1
                unique_persons.add(person_name)
                print(f"  Encoding found for {person_name}")
            else:
                 print(f"  No face found in {os.path.basename(image_path)}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

        images_processed += 1
        progress_bar['value'] = images_processed
        # Allow GUI to update during potentially long training
        window.update_idletasks()
        # window.update() # Might slow down training significantly

    progress_bar['value'] = progress_bar['maximum'] # Ensure it reaches 100%

    if not known_face_names:
         update_status("Training complete, but no faces found/encoded.")
         messagebox.showinfo("Training", "Training complete, but no valid faces were found or encoded in the dataset.")
    else:
        num_persons_trained = len(unique_persons)
        update_status(f"Training complete. Found {encodings_found} encodings for {num_persons_trained} persons. Saving...")
        print(f"Training finished. Found {encodings_found} encodings for {num_persons_trained} persons.")
        if save_trained_model():
             messagebox.showinfo("Training", f"Training complete and model saved.\nData for {num_persons_trained} persons loaded.")
        else:
             messagebox.showerror("Training", "Training completed but failed to save the model.")


def capture_faces(person_name):
    """Captures faces for a new person using Haar Cascade for detection."""
    global cap, add_person_running

    if not person_name or not person_name.strip():
        messagebox.showerror("Error", "Person's name cannot be empty.")
        return

    person_name_clean = person_name.strip()
    person_dir = os.path.join(DATASET_PATH, person_name_clean)
    if not os.path.exists(person_dir):
        try:
            os.makedirs(person_dir)
        except OSError as e:
            messagebox.showerror("Error", f"Failed to create directory for {person_name_clean}: {e}")
            return

    if not initialize_camera():
        return

    update_status(f"Capturing for {person_name_clean}. Look at camera.")
    messagebox.showinfo("Capture Faces", f"Getting ready to capture {CAPTURE_COUNT} faces for {person_name_clean}.\nLook at the camera and press OK.")

    count = 0
    start_time = time.time()
    max_capture_duration = 60 # seconds
    add_person_running = True
    captured_successfully = False

    while count < CAPTURE_COUNT and add_person_running:
        ret, frame = cap.read()
        if not ret:
            update_status("Failed to capture frame.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        display_frame = frame.copy()
        face_detected_this_frame = False

        for (x, y, w, h) in faces:
            # Draw rectangle around the largest detected face for saving
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face_img_path = os.path.join(person_dir, f"{person_name_clean}_{count+1}.jpg")
            # Save the BGR face region
            cv2.imwrite(face_img_path, frame[y:y+h, x:x+w])
            print(f"Captured image {count+1} for {person_name_clean}")
            count += 1
            face_detected_this_frame = True
            update_status(f"Capturing for {person_name_clean}: {count}/{CAPTURE_COUNT}")
            time.sleep(0.2) # Slightly longer delay between captures
            break # Only capture one face per frame

        show_frame(display_frame) # Show frame with detection box

        if time.time() - start_time > max_capture_duration and not face_detected_this_frame:
             update_status("Capture timeout. No face detected.")
             messagebox.showwarning("Capture Timeout", "Could not detect a face for an extended period.")
             break

        window.update_idletasks()
        window.update()

        if not add_person_running:
            break # Check if Stop button was pressed

    # --- Cleanup after capture loop ---
    captured_successfully = (count == CAPTURE_COUNT)
    add_person_running = False
    release_camera()
    clear_video_feed()

    if captured_successfully:
        update_status(f"Capture complete for {person_name_clean}. Please train the model.")
        messagebox.showinfo("Capture Complete", f"Successfully captured {count} images for {person_name_clean}.\nRemember to Train Model.")
    elif count > 0:
         update_status(f"Capture interrupted for {person_name_clean}. Saved {count} images.")
         messagebox.showwarning("Capture Interrupted", f"Capture stopped. Saved {count} images for {person_name_clean}.\nTraining might be less accurate.")
    else:
         update_status(f"Capture failed for {person_name_clean}. No images saved.")
         messagebox.showerror("Capture Failed", f"Could not capture any images for {person_name_clean}. Check lighting/visibility.")
         # Clean up empty directory
         try:
             if not os.listdir(person_dir):
                 os.rmdir(person_dir)
         except OSError:
             pass


def recognize_faces_thread():
    """Handles face recognition in a separate thread."""
    global cap, frame_queue, recognition_stop_event

    if not initialize_camera():
        # Signal GUI that camera failed
        frame_queue.put(None) # Use None as a signal for camera failure
        return

    print("Recognition thread started...")
    frame_count = 0
    process_every_n_frames = 2 # Process every other frame to save CPU

    while not recognition_stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame in recognition thread.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Process frame only periodically
        if frame_count % process_every_n_frames == 0:
            # Resize frame for faster processing (optional but recommended)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces and encodings
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog") # Faster
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            display_frame = frame.copy() # Draw on the original size frame

            # Process detected faces
            for i, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_THRESHOLD)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                # --- Draw rectangle and name (adjust coordinates for original frame size) ---
                top, right, bottom, left = face_locations[i]
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(display_frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                # ---

            # Put the processed frame onto the queue for the GUI thread
            try:
                frame_queue.put(display_frame, block=False) # Non-blocking put
            except queue.Full:
                # If queue is full, just skip this frame to avoid lag
                pass
        else:
            # If not processing, still put the raw frame for smoother display
             try:
                frame_queue.put(frame, block=False)
             except queue.Full:
                pass

        time.sleep(0.01) # Small sleep to prevent busy-waiting

    # --- Thread cleanup ---
    release_camera()
    print("Recognition thread stopped.")


# --- GUI Functions ---

def initialize_camera():
    """Initializes or re-initializes the camera."""
    global cap
    if cap is not None and cap.isOpened():
        # print("Camera already initialized.")
        return True

    release_camera() # Ensure any previous instance is released

    try:
        # Try default camera index first
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW might help on Windows
        if not cap.isOpened():
            print("Trying camera index 1...")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Trying camera index -1...")
            cap = cv2.VideoCapture(-1) # Some systems use -1

        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Check connection/permissions.")
            cap = None
            return False

        # Optional: Set camera properties (might improve performance/stability)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap.set(cv2.CAP_PROP_FPS, 30)
        print("Camera initialized successfully (Index tried: 0, 1, -1).")
        return True
    except Exception as e:
         messagebox.showerror("Camera Error", f"Error initializing camera: {e}")
         cap = None
         return False

def release_camera():
    """Releases the camera resource."""
    global cap
    if cap is not None:
        if cap.isOpened():
            cap.release()
            print("Camera released.")
        cap = None

def show_frame(frame):
    """Converts an OpenCV frame for Tkinter display."""
    global video_label
    if video_label is None:
        return

    try:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        # Consider resizing image for display if needed to fit label
        # img.thumbnail((640, 480)) # Example resize
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    except Exception as e:
        print(f"Error updating frame display: {e}")

def update_video_feed():
    """Periodically checks the queue for new frames and updates the GUI."""
    global frame_queue, video_label, recognition_active, window

    try:
        # Check queue without blocking
        frame = frame_queue.get_nowait()

        if frame is None: # Signal for camera failure
             messagebox.showerror("Camera Error", "Camera failed during recognition.")
             stop_recognition() # Stop the process
             return

        # Display the frame
        show_frame(frame)

    except queue.Empty:
        # No new frame available yet
        pass
    except Exception as e:
        print(f"Error in update_video_feed: {e}")

    # Reschedule if recognition should still be active
    if recognition_active:
        window.after(30, update_video_feed) # Check queue roughly every 30ms

def clear_video_feed():
    """Displays a blank (black) image in the video feed area."""
    global video_label
    if video_label:
        try:
            # Ensure consistent size with camera feed if possible
            w = video_label.winfo_width()
            h = video_label.winfo_height()
            if w < 10 or h < 10:
                w, h = 640, 480 # Default if size not determined
            blank_img = Image.new('RGB', (w, h), (0, 0, 0))
            imgtk = ImageTk.PhotoImage(image=blank_img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk, text="Camera Feed")
        except Exception as e:
            print(f"Error clearing video feed: {e}")


def update_status(message):
    """Updates the status bar."""
    global status_label
    if status_label:
        status_label.config(text=f"Status: {message}")
    print(f"Status: {message}")


# --- Button Click Handlers ---

def on_add_person_click():
    """Handles 'Add Person' button click."""
    if recognition_active:
        messagebox.showwarning("Busy", "Please stop recognition before adding a person.")
        return
    if add_person_running: # Should not happen if GUI is responsive, but check anyway
         messagebox.showwarning("Busy", "Already capturing faces.")
         return

    person_name = simpledialog.askstring("Input", "Enter the person's name:", parent=window)
    if person_name:
        # Disabling buttons during capture
        set_ui_state(capturing=True)
        # Run capture - it now handles its own GUI updates and camera release
        capture_faces(person_name)
        # Re-enable buttons after capture finishes (or fails)
        set_ui_state(capturing=False)
    else:
        update_status("Add person cancelled.")

def on_train_click():
    """Handles 'Train Model' button click."""
    if recognition_active or add_person_running:
        messagebox.showwarning("Busy", "Please stop other operations before training.")
        return

    # Disable buttons during training
    set_ui_state(training=True)
    update_status("Training started...")

    # Run training (consider threading for very large datasets, but usually ok in main thread)
    train_model()

    # Re-enable buttons after training
    set_ui_state(training=False)
    # Status is updated within train_model


def start_recognition():
    """Starts the face recognition thread."""
    global recognition_thread, recognition_stop_event, recognition_active

    if recognition_active:
        messagebox.showinfo("Info", "Recognition is already running.")
        return
    if add_person_running:
        messagebox.showwarning("Busy", "Please wait for face capture to finish.")
        return

    if not known_face_names: # Check if model is loaded/trained
        if not load_trained_model():
             messagebox.showwarning("Recognize Faces", "Model not loaded. Please train the model first.")
             return

    update_status("Starting recognition...")
    set_ui_state(recognizing=True)
    recognition_active = True

    recognition_stop_event.clear() # Ensure the stop flag is clear
    # Clear the queue from any previous runs
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

    # Start the recognition thread
    recognition_thread = threading.Thread(target=recognize_faces_thread, daemon=True)
    recognition_thread.start()

    # Start the GUI update loop
    window.after(100, update_video_feed) # Start checking queue after 100ms

def stop_recognition():
    """Stops the face recognition thread."""
    global recognition_thread, recognition_stop_event, recognition_active

    if not recognition_active:
        # update_status("Recognition not running.")
        return

    update_status("Stopping recognition...")
    recognition_active = False # Signal the update loop to stop rescheduling
    recognition_stop_event.set() # Signal the thread to stop

    # Wait briefly for the thread to finish
    if recognition_thread is not None and recognition_thread.is_alive():
        recognition_thread.join(timeout=1.0) # Wait up to 1 second

    recognition_thread = None

    # Camera is released within the thread, but clear feed here
    clear_video_feed()
    set_ui_state(recognizing=False)
    update_status("Recognition stopped.")

def on_stop_operations_click():
     """Handles the generic stop button."""
     if recognition_active:
         stop_recognition()
     elif add_person_running:
         global add_person_running
         add_person_running = False # Signal capture loop to stop
         update_status("Stopping capture...")
         # Capture loop handles its own cleanup now
     else:
         update_status("No operation running to stop.")


def on_closing():
    """Handles window closing event."""
    print("Closing application...")
    stop_recognition() # Ensure recognition thread is stopped
    global add_person_running
    add_person_running = False # Signal capture loop if running
    release_camera() # Final safety release
    if window:
        window.destroy()

def set_ui_state(capturing=False, training=False, recognizing=False):
    """Enable/disable buttons based on the current state."""
    if capturing or training:
        btn_add.config(state=tk.DISABLED)
        btn_train.config(state=tk.DISABLED)
        btn_start_rec.config(state=tk.DISABLED)
        btn_stop_rec.config(state=tk.DISABLED) # Or enable only Stop if it handles capture/train too
        btn_stop_ops.config(state=tk.NORMAL if capturing else tk.DISABLED) # Only enable stop for capture
    elif recognizing:
        btn_add.config(state=tk.DISABLED)
        btn_train.config(state=tk.DISABLED)
        btn_start_rec.config(state=tk.DISABLED)
        btn_stop_rec.config(state=tk.NORMAL)
        btn_stop_ops.config(state=tk.NORMAL)
    else: # Idle state
        btn_add.config(state=tk.NORMAL)
        btn_train.config(state=tk.NORMAL)
        btn_start_rec.config(state=tk.NORMAL)
        btn_stop_rec.config(state=tk.DISABLED)
        btn_stop_ops.config(state=tk.DISABLED)

# --- Main GUI Setup ---
def create_gui():
    global window, video_label, status_label, progress_bar
    global btn_add, btn_train, btn_start_rec, btn_stop_rec, btn_stop_ops

    window = tk.Tk()
    window.title("Face Recognition System (Enhanced)")

    # --- Video Feed ---
    # Use a frame to better control layout if needed later
    video_frame = tk.Frame(window, bd=2, relief=tk.SUNKEN)
    video_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    video_label = tk.Label(video_frame, text="Camera Feed")
    video_label.pack(fill=tk.BOTH, expand=True)
    clear_video_feed() # Show black background initially

    # --- Progress Bar ---
    progress_bar = ttk.Progressbar(window, orient='horizontal', length=300, mode='determinate')
    progress_bar.pack(pady=5)

    # --- Control Frame ---
    control_frame = tk.Frame(window)
    control_frame.pack(pady=10)

    btn_add = tk.Button(control_frame, text="Add Person", command=on_add_person_click, width=18)
    btn_add.grid(row=0, column=0, padx=5, pady=5)

    btn_train = tk.Button(control_frame, text="Train Model", command=on_train_click, width=18)
    btn_train.grid(row=0, column=1, padx=5, pady=5)

    btn_start_rec = tk.Button(control_frame, text="Start Recognition", command=start_recognition, width=18)
    btn_start_rec.grid(row=1, column=0, padx=5, pady=5)

    btn_stop_rec = tk.Button(control_frame, text="Stop Recognition", command=stop_recognition, width=18, state=tk.DISABLED)
    btn_stop_rec.grid(row=1, column=1, padx=5, pady=5)

    # Added a general stop button for capture (optional, could merge logic)
    btn_stop_ops = tk.Button(window, text="Stop Current Operation", command=on_stop_operations_click, width=25, bg="orange", state=tk.DISABLED)
    btn_stop_ops.pack(pady=5)


    # --- Status Bar ---
    status_label = tk.Label(window, text="Status: Initializing...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Initialization ---
    if not load_face_cascade():
        on_closing()
        return
    if not ensure_dataset_dir():
        on_closing()
        return

    load_trained_model() # Attempt to load existing model on startup
    set_ui_state(recognizing=False) # Set initial button states to idle

    # --- Window Close Handler ---
    window.protocol("WM_DELETE_WINDOW", on_closing)

    update_status("System ready.")
    window.mainloop()


# --- Main Execution ---
if __name__ == "__main__":
    create_gui()