import os
import json
import time
import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Local Directory Paths  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "class_names.json")
MODEL_PATH = os.path.join(BASE_DIR, "best_efficientnet.keras")

# Load Class Names 
try:
    with open(JSON_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    print("Successfully loaded class names!")
except Exception as e:
    print(f"Failed to load class names: {e}")
    CLASS_NAMES = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth_Ulcer", "Tooth Discoloration"]

# Load Model 
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("SUCCESS: The EfficientNet model has been loaded perfectly!\n")
    except Exception as e:
        print(f"ERROR: Found the file, but Keras failed to load it.\nDetails: {e}\n")
else:
    print("ERROR: Model file not found. Please place 'best_efficientnet.keras' in the same folder as this script.\n")

# Enhanced Clinical Advice with Full Descriptions
CLINICAL_INFO = {
    "Calculus": {
        "icon": "🪨",
        "description": "Calculus is hardened dental plaque that forms on teeth when plaque is not properly removed.",
        "symptoms": "Yellow or brown deposits on teeth, gum irritation, bleeding gums.",
        "recommendation": "Professional dental scaling is required. Maintain regular brushing and flossing."
    },
    "Caries": {
        "icon": "🦷",
        "description": "Caries is the destruction of tooth enamel caused by acids produced by bacteria in plaque.",
        "symptoms": "Tooth sensitivity, visible holes or pits in teeth, pain when biting, dark spots.",
        "recommendation": "Visit a dentist for filling or restoration. Use fluoride toothpaste and reduce sugary foods."
    },
    "Gingivitis": {
        "icon": "🔴",
        "description": "Gingivitis is a mild form of gum disease caused by plaque buildup at the gum line, causing inflammation.",
        "symptoms": "Red, swollen gums, bleeding during brushing, receding gum line.",
        "recommendation": "Improve oral hygiene with proper brushing and flossing. Schedule professional cleaning."
    },
    "Hypodontia": {
        "icon": "🔍",
        "description": "Hypodontia is a congenital condition where one or more teeth fail to develop.",
        "symptoms": "Missing teeth, gaps in dental arch, adjacent teeth may shift or tilt.",
        "recommendation": "Consult an orthodontist or dental specialist. Treatment includes bridges, implants, or orthodontics."
    },
    "Mouth_Ulcer": {
        "icon": "⚠️",
        "description": "Mouth ulcers are small, painful lesions that develop in the mouth.",
        "symptoms": "Round or oval sores with red border, white/yellow center, pain while eating or talking.",
        "recommendation": "Most heal within 1-2 weeks. If persistent beyond 3 weeks, seek professional dental advice."
    },
    "Tooth Discoloration": {
        "icon": "🦷",
        "description": "Changes in tooth color due to staining, aging, or damage (extrinsic or intrinsic).",
        "symptoms": "Yellow, brown, or gray tooth color, uneven tooth shade, visible staining.",
        "recommendation": "Professional teeth whitening or veneers may help. Reduce staining foods/beverages."
    }
}

# Helper Function to Create Matplotlib Chart
def create_probability_plot(results, top_class):
    labels = list(results.keys())
    sizes = [v * 100 for v in results.values()]
    
    # Darker Teal gradient
    colors = ['#99F6E4', '#5EEAD4', '#2DD4BF', '#14B8A6', '#0D9488', '#0F766E']
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    bars = ax.barh(labels, sizes, color=colors[:len(labels)], edgecolor='#2DD4BF', linewidth=1, zorder=2)
    
    # Highlight the top predicted class
    top_idx = labels.index(top_class)
    bars[top_idx].set_color('#0F766E')
    bars[top_idx].set_edgecolor('#115E59')
    bars[top_idx].set_linewidth(2)
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', fontsize=10, color='#134E4A', fontweight='600')
    ax.set_title('Confidence Distribution', fontsize=14, fontweight='700', color='#134E4A', pad=15)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10, color='#134E4A', fontweight='700')
    
    # Clean up the chart aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2DD4BF')
    ax.spines['bottom'].set_color('#2DD4BF')
    ax.tick_params(axis='y', colors='#134E4A', labelsize=10, length=0)
    ax.tick_params(axis='x', colors='#134E4A')
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=1, color='#2DD4BF')
    
    plt.tight_layout()
    return fig

# Prediction Function
def predict_image(img):
    global model
    if model is None:
        return (
            "###  Error\nModel is not loaded. Please verify the diagnostic report in your terminal.", 
            {}, 
            None
        )

    if img is None:
        return "### ⏳ Waiting for Image Upload\nPlease upload a clear clinical image to begin analysis.", {}, None

    # Preprocessing
    image = tf.image.resize(img, (224, 224))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image, verbose=0)[0]

    results = {
        CLASS_NAMES[i]: float(prediction[i])
        for i in range(len(CLASS_NAMES))
    }

    top_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Clinical info
    info = CLINICAL_INFO.get(top_class, {
        "icon": "🔬",
        "description": "Condition detected. Please consult a dental professional.",
        "symptoms": "N/A",
        "recommendation": "Please consult a dentist."
    })

    # Build Enhanced Diagnosis Markdown with PROMINENT Confidence Box
    summary_md = f"### {info['icon']} Primary Diagnosis: **{top_class}**\n\n"
    
    
    summary_md += "---\n\n"
    summary_md += f"**📖 Description:**\n{info['description']}\n\n"
    summary_md += f"**🔎 Common Symptoms:**\n{info['symptoms']}\n\n"
    summary_md += f"**💊 Recommendation:**\n{info['recommendation']}"

    # Generate Plot
    plot_fig = create_probability_plot(results, top_class)

    return summary_md, results, plot_fig

# Helper Function to Reset UI
def clear_all():
    initial_md = "### ⏳ Waiting for Image Upload\nPlease upload a clear clinical image to begin analysis."
    return None, initial_md, {}, None

# Medical Teal CSS 
custom_css = """
* {
    box-sizing: border-box;
}

/* Background: Slightly darker */
body {
    background: #ECFDF5;
    min-height: 100vh;
    margin: 0;
    padding: 0;
    color: #134E4A;
}

.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Hero Header - Darker Teal Gradient, Smaller */
.hero-header {
    background: linear-gradient(135deg, #115E59 0%, #0F766E 50%, #0D9488 100%);
    color: white;
    padding: 50px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(255,255,255,0.15) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}

.hero-header h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin: 0 0 12px 0 !important;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    position: relative;
    z-index: 2;
    letter-spacing: -0.5px;
}

.hero-header p {
    font-size: 1.1rem !important;
    color: rgba(255,255,255,0.95) !important;
    max-width: 700px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
    line-height: 1.6;
}

.content-wrapper {
    max-width: 1400px;
    margin: 0 auto;
    padding: 40px;
}

/* Features Badges - Darker borders */
.features-badges {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: -30px auto 50px auto;
    position: relative;
    z-index: 10;
    flex-wrap: wrap;
}

.feature-badge {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 18px 26px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.15);
    border: 1px solid #2DD4BF;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.feature-badge:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 28px rgba(15, 118, 110, 0.25);
    border-color: #0F766E;
}

.feature-badge-icon { font-size: 2rem; }
.feature-badge-text { display: flex; flex-direction: column; }
.feature-badge-title { font-size: 1.05rem; font-weight: 700; color: #134E4A; }
/*  Changed from #2DD4BF to #0D9488 for better visibility */
.feature-badge-desc { font-size: 0.85rem; color: #0D9488; margin-top: 2px; }

/* Workspace Container - Darker borders */
.workspace-container {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 40px;
    border: 1px solid #2DD4BF;
    box-shadow: 0 4px 16px rgba(15, 118, 110, 0.1);
}

.workspace-header { text-align: center; margin-bottom: 35px; }
.workspace-header h2 { 
    color: #134E4A !important; 
    font-size: 1.9rem !important; 
    font-weight: 800 !important; 
    margin: 0 0 10px 0 !important; 
}
/* Changed from #2DD4BF to #0D9488 for better visibility */
.workspace-header p { 
    color: #0D9488 !important; 
    font-size: 1.05rem !important; 
    margin: 0 !important; 
}

.workspace-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
}

/* Workspace Panels - Darker borders */
.workspace-panel {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #2DD4BF;
}

.panel-title {
    color: #134E4A !important;
    font-weight: 800 !important;
    font-size: 1.25rem !important;
    margin: 0 0 20px 0 !important;
    padding-bottom: 15px;
    border-bottom: 2px solid #2DD4BF;
}

/* Buttons - Darker Primary: #0F766E, Hover: #115E59 */
.btn-primary {
    background: #0F766E !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 15px 28px !important;
    border-radius: 10px !important;
    width: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.35) !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    background: #115E59 !important;
    box-shadow: 0 8px 20px rgba(15, 118, 110, 0.45) !important;
}

.btn-secondary {
    background: #FFFFFF !important;
    color: #134E4A !important;
    border: 2px solid #2DD4BF !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 13px 24px !important;
    width: 100%;
    transition: all 0.2s ease !important;
}
.btn-secondary:hover {
    background: #ECFDF5 !important;
    color: #134E4A !important;
    border-color: #0F766E !important;
}

/* Diagnosis Box - Darker border */
.diagnosis-box {
    background: #FFFFFF !important;
    border: 2px solid #0F766E !important;
    border-radius: 14px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 2px 8px rgba(15, 118, 110, 0.12) !important;
}

.diagnosis-box h3 {
    color: #134E4A !important;
}

/* Analytics Section - Darker borders */
.analytics-container {
    margin-top: 30px;
    background: #FFFFFF;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #2DD4BF;
}

.analytics-title {
    color: #134E4A !important;
    font-weight: 800 !important;
    font-size: 1.25rem !important;
    margin: 0 0 20px 0 !important;
    padding-bottom: 15px;
    border-bottom: 2px solid #2DD4BF;
}

/* Footer -  Changed from #2DD4BF to #0D9488 for better visibility */
.footer-text {
    text-align: center;
    color: #0D9488;
    font-size: 0.9rem;
    padding: 35px 20px;
    border-top: 1px solid #2DD4BF;
    margin-top: 30px;
}
.footer-text strong { color: #134E4A; font-weight: 800; }

/* Text color override */
.gradio-container p, 
.gradio-container li,
.gradio-container span {
    color: #134E4A;
}

@media (max-width: 768px) {
    .workspace-grid { grid-template-columns: 1fr; }
    .features-badges { flex-direction: column; align-items: center; }
    .hero-header h1 { font-size: 1.8rem !important; }
    .content-wrapper { padding: 20px; }
}
"""

# Build the Web Interface Layout
with gr.Blocks(css=custom_css) as demo:
    
    gr.HTML(
        """
        <div class="hero-header">
            <h1>🔬 AI-Based Oral Diseases Classification</h1>
            <p>Intelligent Clinical Evaluation & Oral Disease Detection System</p>
        </div>
        """
    )

    with gr.Column(elem_classes="content-wrapper"):
        
        gr.HTML(
            """
            <div class="features-badges">
                <div class="feature-badge">
                    <div class="feature-badge-icon">🎯</div>
                    <div class="feature-badge-text">
                        <div class="feature-badge-title">Clinical Precision</div>
                        <div class="feature-badge-desc">Expert-grade detection</div>
                    </div>
                </div>
                <div class="feature-badge">
                    <div class="feature-badge-icon">⚡</div>
                    <div class="feature-badge-text">
                        <div class="feature-badge-title">Fast Analysis</div>
                        <div class="feature-badge-desc">Real-time results</div>
                    </div>
                </div>
                <div class="feature-badge">
                    <div class="feature-badge-icon">📊</div>
                    <div class="feature-badge-text">
                        <div class="feature-badge-title">Data Driven</div>
                        <div class="feature-badge-desc">Visual analytics</div>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Column(elem_classes="workspace-container"):
            
            gr.HTML(
                """
                <div class="workspace-header">
                    <h2>Start Your Analysis</h2>
                    <p>Upload a clinical image to get instant AI-powered diagnostics</p>
                </div>
                """
            )

            with gr.Row(elem_classes="workspace-grid"):
                
                # Input
                with gr.Column(elem_classes="workspace-panel"):
                    gr.HTML('<div class="panel-title">📤 Upload Clinical Image</div>')
                    image_input = gr.Image(
                        label="Drop or click to upload", 
                        type="numpy",
                        height=380
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Analyze Case", elem_classes="btn-primary")
                        clear_btn = gr.Button("Clear", elem_classes="btn-secondary")

                # Output
                with gr.Column(elem_classes="workspace-panel"):
                    gr.HTML('<div class="panel-title">📊 Diagnostic Results</div>')
                    
                    summary_output = gr.Markdown(
                        value="### ⏳ Waiting for Image Upload\nPlease upload a clear clinical image to begin analysis.",
                        elem_classes="diagnosis-box"
                    )
                    
                    label_output = gr.Label(
                        num_top_classes=6, 
                        label="Quick Probability View"
                    )

            # Analytics Section with Plot
            with gr.Column(elem_classes="analytics-container"):
                gr.HTML('<div class="analytics-title">📈 Confidence Distribution Analytics</div>')
                plot_output = gr.Plot(label="Detailed Probability Breakdown")

    gr.HTML(
        """
        <div class="footer-text">
            <p><strong>🔬 AI-Based Oral Diseases Classification</strong> — Powered by Deep Learning & Computer Vision</p>
            <p style="font-size: 0.85rem; margin-top: 10px; color: #0D9488;">
                Disclaimer: This tool is for educational and screening purposes only. Always consult a certified dentist for final diagnosis.
            </p>
        </div>
        """
    )

    # Link Actions
    submit_btn.click(
        fn=predict_image, 
        inputs=image_input, 
        outputs=[summary_output, label_output, plot_output]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[image_input, summary_output, label_output, plot_output]
    )

# Launch Application
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        share=False
    )
