import os
import sys
import numpy as np
from PIL import Image
import gradio as gr
import torch
import time
from contextlib import nullcontext

# Configurer les chemins nécessaires
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Configuration optimisée pour RTX 4060
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_VERBOSE"] = "True"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Optimisation CUDA supplémentaire pour RTX 4060
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()  # Vider le cache CUDA au démarrage

# Fonction pour détecter les capacités GPU et ajuster les paramètres
def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return gpu_name, gpu_memory_total
    return "CPU", 0

def detect_gpu_capacity():
    """Détecte les capacités GPU et retourne les paramètres optimaux"""
    if not torch.cuda.is_available():
        return {
            "width": 384, 
            "height": 512, 
            "precision": "float32", 
            "batch_size": 1,
            "steps_scale": 0.5
        }
    
    gpu_name, gpu_memory_total = get_gpu_info()
    
    # Optimisations spécifiques pour RTX 4060
    if "RTX 4060" in gpu_name:
        if gpu_memory_total >= 8.0:  # 8GB VRAM
            return {
                "width": 768, 
                "height": 1024, 
                "precision": "bfloat16", 
                "batch_size": 1,
                "steps_scale": 0.8
            }
        else:  # < 8GB VRAM
            return {
                "width": 512, 
                "height": 768, 
                "precision": "float16", 
                "batch_size": 1,
                "steps_scale": 0.7
            }
    # Autres GPU NVIDIA avec beaucoup de VRAM
    elif gpu_memory_total >= 12.0:  # >= 12GB VRAM
        return {
            "width": 768, 
            "height": 1024, 
            "precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float16", 
            "batch_size": 1,
            "steps_scale": 0.8
        }
    # GPU avec VRAM moyenne
    elif gpu_memory_total >= 8.0:  # >= 8GB VRAM
        return {
            "width": 768, 
            "height": 1024, 
            "precision": "float16", 
            "batch_size": 1,
            "steps_scale": 0.7
        }
    # GPU avec peu de VRAM
    elif gpu_memory_total >= 4.0:  # >= 4GB VRAM
        return {
            "width": 512, 
            "height": 768, 
            "precision": "float16", 
            "batch_size": 1,
            "steps_scale": 0.6
        }
    else:
        return {
            "width": 384, 
            "height": 512, 
            "precision": "float32", 
            "batch_size": 1,
            "steps_scale": 0.5
        }

# Paramètres GPU optimaux basés sur la détection
gpu_params = detect_gpu_capacity()

print("\n===== DÉMARRAGE DE LEFFA ULTRA (MODE SIMPLIFIÉ) =====")
print("L'application va démarrer et sera accessible à l'adresse http://127.0.0.1:7860")
print("Ce mode est optimisé pour l'essayage virtuel avec performances maximales sur GPU RTX 4060.")
print(f"Résolution adaptée dynamiquement: {gpu_params['width']}x{gpu_params['height']}")
print(f"Précision numérique: {gpu_params['precision']}")

# Initialisation de l'interface Gradio simplifiée
try:
    print("Création de l'interface Gradio simplifiée...")
    
    with gr.Blocks(title="Leffa Ultra - Mode Simplifié", css="footer {visibility: hidden}") as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 800px; margin: 0 auto;">
                <div>
                    <h1 style="font-weight: 900; font-size: 3rem; margin: 0.5rem 0;">🧥 Leffa Ultra - Mode Simplifié 🧥</h1>
                    <h2 style="font-weight: 450; font-size: 1rem; margin: 0.5rem 0;">Optimisé pour RTX 4060</h2>
                </div>
                <p style="margin-bottom: 10px; font-size: 94%">
                    Cette version simplifiée de Leffa Ultra est en cours de configuration pour votre système.
                </p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚠️ Mode Maintenance")
                gr.Markdown("""
                L'application Leffa Ultra est actuellement en mode maintenance.
                
                Nous rencontrons des problèmes avec certaines dépendances qui nécessitent des outils de compilation C++ (Visual Studio Build Tools).
                
                Pour résoudre ce problème, vous pouvez:
                
                1. Installer Visual Studio Build Tools avec le composant "Outils de développement C++"
                2. Utiliser une version précompilée de l'application
                
                L'équipe travaille activement à résoudre ce problème pour vous offrir une expérience optimale avec votre GPU RTX 4060.
                """)
                
                gpu_info = gr.Markdown(f"""
                **Informations GPU**
                - Dispositif: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
                - Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB si disponible
                - Résolution configurée: {gpu_params['width']}x{gpu_params['height']}
                - Précision configurée: {gpu_params['precision']}
                """)
    
    print("\n===== INTERFACE LEFFA SIMPLIFIÉE PRÊTE =====")
    demo.launch(show_api=False)
    
except Exception as e:
    print(f"Erreur lors de l'initialisation: {e}")
    import traceback
    traceback.print_exc()
