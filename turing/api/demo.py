import csv
from datetime import datetime
import os

import gradio as gr

# ---IMPORTS ---
try:
    from turing.modeling.models.codeBerta import CodeBERTa
    from turing.modeling.predict import ModelInference
except ImportError as e:
    print(f"WARNING: Error importing real modules: {e}")
    class CodeBERTa: 
        pass
    class ModelInference: 
        pass

# --- CONFIGURATION ---
FEEDBACK_FILE = "reports/feedback/feedback_data.csv"

LABELS_MAP = {
    "java": ["summary", "Ownership", "Expand", "usage", "Pointer", "deprecation", "rational"],
    "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
    "pharo": ["Keyimplementationpoints", "Example", "Responsibilities", "Intent", "Keymessages", "Collaborators"],
}

# --- CSS ---
CSS = """
:root {
    --bg-primary: #fafaf9; --bg-secondary: #ffffff; --border-color: #e5e7eb;
    --text-primary: #1f2937; --text-secondary: #6b7280; --accent-bg: #f3f4f6;
    --primary-btn: #ea580c; --primary-btn-hover: #c2410c;
}
.dark, body.dark, .gradio-container.dark {
    --bg-primary: #0f172a; --bg-secondary: #1e293b; --border-color: #374151;
    --text-primary: #f3f4f6; --text-secondary: #9ca3af; --accent-bg: #334155;
}
body, .gradio-container {
    background-color: var(--bg-primary) !important; color: var(--text-primary) !important;
    font-family: 'Segoe UI', system-ui, sans-serif; transition: background 0.3s, color 0.3s;
}
.compact-header {
    display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border-color); margin-bottom: 2rem;
    background-color: var(--bg-secondary); flex-wrap: wrap; gap: 1rem; border-radius: 0 0 12px 12px;
}
.input-card, .output-card {
    background-color: var(--bg-secondary); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.header-left { display: flex; align-items: center; gap: 1.5rem; }
.logo-icon {
    height: 55px; width: auto; padding: 0; background-color: transparent;
    border: none; box-shadow: none; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.logo-icon svg { height: 100%; width: auto; fill: var(--primary-btn); }
.title-group { display: flex; flex-direction: column; }
.main-title { font-size: 1.6rem; font-weight: 800; margin: 0; line-height: 1.1; color: var(--text-primary); letter-spacing: -0.5px; }
.subtitle { font-size: 0.95rem; color: var(--text-secondary); margin: 0; font-weight: 400; }
.section-title { font-weight: 600; color: var(--text-primary); margin-bottom: 1rem; }
.header-right { flex: 1; display: flex; justify-content: flex-end; align-items: center; min-width: 250px; }
.dev-note-container {
    background-color: var(--accent-bg); border: 1px solid var(--border-color); border-radius: 16px;
    width: 520px; height: 64px; display: flex; align-items: center; justify-content: flex-start; padding: 0 24px; gap: 1rem;
}
.dev-note-container:hover { border-color: var(--primary-btn); }
.dev-icon { font-size: 1.4rem; background: transparent !important; border: none !important; display: flex; align-items: center; flex-shrink: 0; }
.dev-text {
    font-family: 'Courier New', monospace; font-size: 0.95rem; color: var(--text-secondary);
    transition: opacity 1.5s ease; white-space: normal; line-height: 1.2; text-align: left;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.dev-text.hidden { opacity: 0; }
.feedback-section { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px dashed var(--border-color); }
.feedback-title { font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 0.8rem; }
.gr-button-primary { background: var(--primary-btn) !important; border: none !important; color: white !important; }
.gr-button-primary:hover { background: var(--primary-btn-hover) !important; }
.gr-button-secondary { background: var(--bg-primary) !important; border: 1px solid var(--border-color) !important; color: var(--text-primary) !important; }
.gr-box, .gr-input, .gr-dropdown { background: var(--bg-primary) !important; border-color: var(--border-color) !important; }
#result-box textarea {
    font-size: 1.25rem; font-weight: 700; text-align: center; color: var(--primary-btn);
    background-color: transparent; border: none; overflow: hidden !important; resize: none; white-space: normal; line-height: 1.4;
}
"""

# --- JAVASCRIPT ---
JS_LOADER = """
() => {
    const notes = [
        "Yes, even Pharo. Don’t ask why.", 
        "Is ‘deprecated’ significant? Asking for a friend.",
        "Technical debt is just future-me's problem.",
        "Comment first, code later. Obviously.",
        "If it works, don't touch it.",
        "Fixing bugs created by previous-me.",
        "Legacy code: don't breathe on it.",
        "Documentation is a love letter to your future self.",
        "It works on my machine!",
        "404: Motivation not found.",
        "Compiling... please hold."
    ];
    let idx = 0;
    function rotateNotes() {
        const textEl = document.getElementById('dev-note-text');
        if (!textEl) { setTimeout(rotateNotes, 500); return; }
        textEl.classList.add('hidden');
        setTimeout(() => {
            idx = (idx + 1) % notes.length;
            textEl.innerText = notes[idx];
            textEl.classList.remove('hidden');
        }, 1500);
    }
    setInterval(rotateNotes, 10000);
}
"""

# --- UTILITIES ---
def load_svg_content(filename="logo_header.svg"):
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    target_path = os.path.join(base_path, "..", "..", "reports", "figures", filename)
    target_path = os.path.normpath(target_path)
    
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"[WARNING] Logo not found in: {target_path}")
        return "<span style='color: var(--primary-btn); font-weight:bold;'>CCC</span>"

def save_feedback_to_csv(text, language, predicted, suggested):
    if not text: 
        return "No data."
    try:
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        file_exists = os.path.isfile(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Input_Text", "Language", "Model_Prediction", "User_Correction"])
            
            pred_label = predicted
            if isinstance(predicted, dict):
                pred_label = max(predicted, key=predicted.get) if predicted else "Unknown"

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                text.strip(),
                language,
                pred_label,
                suggested
            ])
        return "Feedback saved successfully!"
    except Exception as e:
        return f"Error saving feedback: {str(e)}"

# --- SYNTAX VALIDATION LOGIC ---
def is_valid_syntax(text: str, language: str) -> bool:
    """
    Validates if the text follows the basic comment syntax for the given language.
    """
    text = text.strip()
    if not text:
        return False

    if language == "java":
        # Supports: // comment OR /* comment */
        return text.startswith("//") or (text.startswith("/*") and text.endswith("*/"))
    
    elif language == "python":
        # Supports: # comment OR """ docstring """ OR ''' docstring '''
        return text.startswith("#") or \
               (text.startswith('"""') and text.endswith('"""')) or \
               (text.startswith("'''") and text.endswith("'''"))
    
    elif language == "pharo":
        # Supports: " comment "
        return text.startswith('"') and text.endswith('"')
    
    return True

# --- MAIN DEMO ---
def create_demo(inference_engine: ModelInference):
    
    def classify_comment(text: str, language: str):
        """
        Calls the inference engine only if syntax is valid.
        """
        if not text: 
            return None
        
        # SYNTAX CHECK
        if not is_valid_syntax(text, language):
            error_msg = "Error: Invalid Syntax."
            if language == "java":
                error_msg += " Java comments must start with '//' or be enclosed in '/* ... */'."
            elif language == "python":
                error_msg += " Python comments must start with '#' or use docstrings ('\"\"\"' / \"'''\")."
            elif language == "pharo":
                error_msg += " Pharo comments must be enclosed in double quotes (e.g., \"comment\")."
            return error_msg

        # INFERENCE
        try:
            _, labels, _, _ = inference_engine.predict_payload(
                texts=[text], 
                language=language
            )
            
            if labels and len(labels) > 0:
                first_prediction = labels[0][0]
                if isinstance(first_prediction, (list, tuple)):
                    return first_prediction[0] 
                else:
                    return str(first_prediction)
            
            return "Unknown: Low confidence."

        except Exception as e:
            print(f"Prediction Error: {e}")
            return f"System Error: Failed to process request for '{language}'."

    def update_dropdown(language):
        choices = LABELS_MAP.get(language, [])
        return gr.Dropdown(choices=choices, value=None, interactive=True)
    
    def clear_all():
        return (None, "java", "", gr.Dropdown(choices=LABELS_MAP["java"], value=None, interactive=True), "")

    logo_svg = load_svg_content("logo_header.svg")

    with gr.Blocks(title="Code Comment Classifier") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        
        # --- HEADER ---
        gr.HTML(f"""
            <div class="compact-header">
                <div class="header-left">
                    <div class="logo-icon">{logo_svg}</div>
                    <div class="title-group">
                        <h1 class="main-title">Code Comment Classifier</h1>
                        <p class="subtitle">for Java, Python & Pharo</p>
                    </div>
                </div>
                <div class="header-right">
                    <div class="dev-note-container">
                        <span class="dev-icon" style="color: var(--primary-btn);">💭</span>
                        <span id="dev-note-text" class="dev-text">Initializing...</span>
                    </div>
                </div>
            </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="input-card"><div class="section-title">📝 Input Source</div></div>')
                input_text = gr.Textbox(label="Code Comment", lines=8, show_label=False, placeholder="Enter code comment here...")
                with gr.Row():
                    input_lang = gr.Dropdown(["java", "python", "pharo"], label="Language", value="java", scale=2)
                    submit_btn = gr.Button("⚡ Classify", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary", size="sm")

            with gr.Column():
                gr.HTML('<div class="output-card"><div class="section-title">📊 Classification Result</div></div>')
                output_tags = gr.Textbox(
                    label="Predicted Category", 
                    show_label=False, 
                    elem_id="result-box",
                    interactive=False,
                    lines=2
                )
                
                gr.HTML('<div class="feedback-section"><div class="feedback-title">🛠️ Help Improve the Model</div></div>')
                with gr.Row():
                    correction_dropdown = gr.Dropdown(
                        choices=LABELS_MAP["java"], 
                        label="Correct Label", 
                        show_label=False, 
                        container=False, 
                        scale=3, 
                        interactive=True
                    )
                    feedback_btn = gr.Button("📤 Save Feedback", variant="secondary", scale=1)
                feedback_msg = gr.Markdown("", show_label=False)

        gr.Examples(
            examples=[
                ["/** Validates the user session token. */", "java"],
                ["# Retry logic for DB connection.", "python"],
                ['"Manages the network connection lifecycle."', "pharo"]
            ],
            inputs=[input_text, input_lang],
            label="Quick Examples"
        )

        input_lang.change(fn=update_dropdown, inputs=input_lang, outputs=correction_dropdown)
        submit_btn.click(fn=classify_comment, inputs=[input_text, input_lang], outputs=[output_tags])
        feedback_btn.click(fn=save_feedback_to_csv, inputs=[input_text, input_lang, output_tags, correction_dropdown], outputs=[feedback_msg])
        clear_btn.click(fn=clear_all, inputs=None, outputs=[input_text, input_lang, output_tags, correction_dropdown, feedback_msg])

        demo.load(None, js=JS_LOADER)

    return demo