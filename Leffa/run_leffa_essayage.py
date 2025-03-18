import os
import sys
import numpy as np
from PIL import Image
import gradio as gr

# Configurer les chemins nécessaires
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '3rdparty'))
sys.path.append(os.path.join(current_dir, '3rdparty/densepose'))
sys.path.append(os.path.join(current_dir, '3rdparty/SCHP'))
sys.path.append(os.path.join(current_dir, '3rdparty/detectron2'))

# Configuration pour économiser la mémoire
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_VERBOSE"] = "True"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Forcément utiliser CPU si mémoire limitée
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Forcer l'utilisation du CPU

print("\n===== DÉMARRAGE DE LEFFA (MODE ESSAYAGE VIRTUEL UNIQUEMENT) =====")
print("L'application va démarrer et sera accessible à l'adresse http://127.0.0.1:7860")
print("Ce mode est optimisé pour l'essayage virtuel uniquement.")

# Import des modules nécessaires pour l'essayage virtuel
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Téléchargement des checkpoints (uniquement ceux nécessaires pour l'essayage virtuel)
print("Téléchargement des modèles minimaux requis...")
snapshot_download(repo_id="franciszzj/Leffa", 
                  local_dir="./ckpts", 
                  allow_patterns=[
                      "examples/**",
                      "stable-diffusion-inpainting/**",
                      "densepose/**", 
                      "schp/**", 
                      "humanparsing/**", 
                      "openpose/**",
                      "virtual_tryon.pth"
                  ])

class LeffaEssayagePredictor(object):
    def __init__(self):
        print("Initialisation des prédicteurs (mode léger)...")
        
        # Chargement des modèles essentiels uniquement
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        # Uniquement le modèle HD pour l'essayage virtuel
        print("Chargement du modèle d'essayage virtuel...")
        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float32",  # Utiliser float32 pour éviter les problèmes de mémoire
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

    def leffa_predict_vt(self, 
                         src_image_path, 
                         ref_image_path, 
                         ref_acceleration=False, 
                         step=30,  # Réduit à 30 pour économiser la mémoire (au lieu de 50)
                         scale=2.5, 
                         seed=42, 
                         vt_model_type="viton_hd", 
                         vt_garment_type="upper_body", 
                         vt_repaint=False, 
                         preprocess_garment=False):
        
        print(f"Traitement des images: {src_image_path} et {ref_image_path}")
        
        # Ouvrir et redimensionner l'image source
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        # Prétraiter l'image du vêtement si nécessaire
        if preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("L'image du vêtement doit être au format PNG lorsque le prétraitement est activé.")
        else:
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, 768, 1024)
        src_image_array = np.array(src_image)
        
        # Préparation des données pour l'essayage virtuel
        src_image = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        
        mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
        mask = mask.resize((768, 1024))
        
        # Préparation de DensePose
        src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        src_image_seg = Image.fromarray(src_image_seg_array)
        densepose = src_image_seg
        
        # Transformation des données
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        
        # Inférence
        output = self.vt_inference_hd(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)

# Initialisation du prédicteur
try:
    print("Création du prédicteur Leffa (mode essayage uniquement)...")
    leffa_predictor = LeffaEssayagePredictor()
    
    # Préparation des exemples
    example_dir = "./ckpts/examples"
    person1_images = list_dir(f"{example_dir}/person1")
    garment_images = list_dir(f"{example_dir}/garment")
    
    print("Création de l'interface Gradio...")
    
    # Interface Gradio simplifiée
    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink)) as demo:
        gr.Markdown("## Leffa - Essayage Virtuel (Mode Économie Mémoire)")
        gr.Markdown("Ce mode utilise uniquement la fonctionnalité d'essayage virtuel pour économiser de la mémoire.")
        
        with gr.Row():
            with gr.Column():
                src_image = gr.Image(type="filepath", label="Image de la personne")
                ref_image = gr.Image(type="filepath", label="Image du vêtement")
                
                with gr.Accordion("Paramètres avancés", open=False):
                    vt_model_type = gr.Radio(
                        ["viton_hd"],
                        value="viton_hd",
                        label="Modèle d'essayage virtuel",
                    )
                    
                    vt_garment_type = gr.Radio(
                        ["upper_body", "lower_body", "dresses"],
                        value="upper_body",
                        label="Type de vêtement",
                    )
                    
                    preprocess_garment = gr.Checkbox(
                        value=False,
                        label="Prétraiter l'image du vêtement (PNG transparent uniquement)",
                    )
                    
                    ref_acceleration = gr.Checkbox(
                        value=False, 
                        label="Accélération référence"
                    )
                    
                    vt_repaint = gr.Checkbox(
                        value=False, 
                        label="Repeindre"
                    )
                    
                    step = gr.Slider(
                        1, 50, value=20, step=1, 
                        label="Nombre d'étapes d'inférence"
                    )
                    
                    scale = gr.Slider(
                        1, 10, value=2.5, step=0.5, 
                        label="Échelle de guidage"
                    )
                    
                    seed = gr.Number(
                        value=42, 
                        label="Graine aléatoire"
                    )
                
                predict_btn = gr.Button("Essayer le vêtement", variant="primary")
                
            with gr.Column():
                result_image = gr.Image(label="Résultat")
                mask_image = gr.Image(label="Masque", visible=False)
                densepose_image = gr.Image(label="DensePose", visible=False)
        
        # Exemples
        gr.Examples(
            examples=[
                [person1_images[0], garment_images[0], "viton_hd", "upper_body", False, False, False, 20, 2.5, 42],
                [person1_images[1], garment_images[1], "viton_hd", "upper_body", False, False, False, 20, 2.5, 42],
            ],
            inputs=[
                src_image, ref_image, vt_model_type, vt_garment_type, preprocess_garment,
                ref_acceleration, vt_repaint, step, scale, seed
            ],
        )
        
        # Fonction de prédiction
        predict_btn.click(
            fn=leffa_predictor.leffa_predict_vt,
            inputs=[
                src_image, ref_image, ref_acceleration, step, scale, seed,
                vt_model_type, vt_garment_type, vt_repaint, preprocess_garment
            ],
            outputs=[result_image, mask_image, densepose_image],
        )
    
    # Lancement de l'interface
    print("\n===== INTERFACE LEFFA PRÊTE (MODE ESSAYAGE UNIQUEMENT) =====")
    print("Démarrage de l'interface sur http://127.0.0.1:7860")
    
    demo.queue(max_size=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        max_threads=1,      # Réduire le nombre de threads
        prevent_thread_lock=True
    )
    
except Exception as e:
    print(f"ERREUR: {str(e)}")
    import traceback
    traceback.print_exc()
