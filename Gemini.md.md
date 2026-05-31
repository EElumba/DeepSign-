# Project Overview: Real-Time Sign Language Translator

**Context:** This project is being built during a 24-hour hackathon. Speed, simplicity, and a functional demo are the highest priorities. Avoid over-engineering, complex abstractions, or boilerplate code. 

**Goal:** A purely client-side web application that uses the user's webcam to recognize hand gestures (sign language) in real-time and translates them into text on the screen.

---

## Tech Stack
*   **Frontend Framework:** React 18+ (initialized via Vite)
*   **Styling:** Tailwind CSS (use utility classes exclusively, avoid custom CSS files unless absolutely necessary)
*   **Camera Integration:** `react-webcam`
*   **Machine Learning:** `@mediapipe/tasks-vision` (Google MediaPipe)
*   **Hosting:** Vercel (Static export)

---

## Architectural Rules (CRITICAL)
1.  **NO BACKEND:** Do not suggest adding Node.js, Express, Python, WebSockets, or any backend logic. The entire ML pipeline runs locally in the browser via WebAssembly to guarantee zero latency and easy deployment.
2.  **MediaPipe Model Location:** The pre-trained model (`gesture_recognizer.task`) is stored in the `public/` directory and must be fetched as a static asset (e.g., `"/gesture_recognizer.task"`).
3.  **Performance:** Video processing must occur within a `requestAnimationFrame` loop to prevent blocking the main UI thread. 

---

## Folder Structure
```text
sign-language-app/
├── public/
│   └── gesture_recognizer.task    # Static MediaPipe model
├── src/
│   ├── components/
│   │   ├── WebcamFeed.jsx         # Renders <Webcam/> and handles the video ref
│   │   └── GestureDisplay.jsx     # Renders the translated text UI
│   ├── utils/
│   │   └── mediapipe.js           # Initializes and runs the MediaPipe Vision Task
│   ├── App.jsx                    # Main state management (sentences, current sign)
│   ├── main.jsx                   # React entry point
│   └── index.css                  # Tailwind directives