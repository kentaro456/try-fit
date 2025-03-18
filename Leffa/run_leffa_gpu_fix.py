import os
import sys
import numpy as np
from PIL import Image
import gradio as gr
import torch

# Configurer les chemins nécessaires
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '3rdparty'))
sys.path.append(os.path.join(current_dir, '3rdparty/densepose'))
sys.path.append(os.path.join(current_dir, '3rdparty/SCHP'))
sys.path.append(os.path.join(current_dir, '3rdparty/detectron2'))

# Configuration optimisée pour RTX 4060
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_VERBOSE"] = "True"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Optimisation CUDA supplémentaire pour RTX 4060
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()  # Vider le cache CUDA au démarrage

# Patch les fonctions non compatibles avec CUDA
import torchvision
from torchvision.ops.boxes import _box_inter_union, remove_small_boxes

def patched_batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    Version modifiée qui évite l'erreur CUDA en utilisant le CPU uniquement pour la fonction NMS.
    """
    # Déplacer les données vers le CPU pour l'opération NMS
    cpu_boxes = boxes.cpu()
    cpu_scores = scores.cpu()
    cpu_idxs = idxs.cpu()
    
    # Appliquer NMS sur CPU
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # strategy: in order to perform NMS independently per class,
    # we add an offset to each box based on its class. This offset ensures that
    # boxes from different classes do not overlap
    max_coordinate = cpu_boxes.max()
    offsets = cpu_idxs.to(torch.float) * (max_coordinate + 1)
    boxes_for_nms = cpu_boxes + offsets[:, None]
    
    # Utilise la version CPU de NMS
    cpu_keep = torch.ops.torchvision.nms(boxes_for_nms, cpu_scores, iou_threshold)
    
    # Renvoie les résultats sur le même device que l'entrée
    return cpu_keep.to(boxes.device)

# Sauvegarder la référence à la fonction originale AVANT de la patcher
original_roi_align = torch.ops.torchvision.roi_align

# Patch pour l'opération roi_align qui n'est pas compatible avec CUDA
def patched_roi_align(input, rois, spatial_scale, output_size_h, output_size_w, sampling_ratio, aligned):
    """
    Version modifiée de roi_align qui utilise le CPU puis renvoie les résultats sur le même device que l'entrée.
    """
    original_device = input.device
    
    # Si déjà sur CPU, pas besoin de convertir
    if original_device.type == 'cpu':
        return original_roi_align(input, rois, spatial_scale, output_size_h, output_size_w, sampling_ratio, aligned)
    
    # Déplacer les données vers le CPU
    cpu_input = input.cpu()
    cpu_rois = rois.cpu()
    
    # Effectuer l'opération roi_align sur CPU avec la fonction originale
    result = original_roi_align(
        cpu_input, 
        cpu_rois, 
        spatial_scale, 
        output_size_h, 
        output_size_w, 
        sampling_ratio, 
        aligned
    )
    
    # Renvoyer le résultat sur le même device que l'entrée
    return result.to(original_device)

# Patcher les fonctions dans torchvision
torchvision.ops.boxes._batched_nms_coordinate_trick = patched_batched_nms

# Rediriger les appels à roi_align vers notre version patchée
torch.ops.torchvision.roi_align = patched_roi_align

print("\n===== DÉMARRAGE DE LEFFA ULTRA (MODE ESSAYAGE VIRTUEL OPTIMISÉ) =====")
print("L'application va démarrer et sera accessible à l'adresse http://127.0.0.1:7860")
print("Ce mode est optimisé pour l'essayage virtuel avec performances maximales sur GPU RTX 4060.")

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
        print("Initialisation des prédicteurs (mode GPU optimisé)...")
        
        # Vérification de la disponibilité du GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du dispositif: {self.device}")
        
        # Afficher les informations GPU
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Mémoire GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
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
            dtype="float16",  # Utiliser float16 pour le GPU
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

    def leffa_predict_vt(self, 
                         src_image_path, 
                         ref_image_path, 
                         ref_acceleration=False, 
                         step=20,  # Réduit à 20 pour plus de rapidité
                         scale=3.0,  # Augmenté à 3.0 pour meilleure qualité
                         seed=42, 
                         vt_model_type="viton_hd", 
                         vt_garment_type="upper_body", 
                         vt_repaint=False, 
                         preprocess_garment=False,
                         optimise_performance=True):  # Nouveau paramètre pour optimiser les performances
        
        print(f"Traitement des images: {src_image_path} et {ref_image_path}")
        
        # Choisir la taille en fonction du type de vêtement et de l'optimisation
        width, height = (768, 1024)  # Taille standard HD
        if optimise_performance:
            # Réduire la taille des images pour les performances
            width, height = (512, 768) if vt_garment_type in ["upper_body", "lower_body"] else (640, 832)
        
        # Optimisation de la mémoire GPU
        torch.cuda.empty_cache()
        
        # Ouvrir et redimensionner l'image source
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, width, height)

        # Prétraiter l'image du vêtement si nécessaire
        if preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("L'image du vêtement doit être au format PNG lorsque le prétraitement est activé.")
        else:
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, width, height)
        src_image_array = np.array(src_image)
        
        # Préparation des données pour l'essayage virtuel
        src_image = src_image.convert("RGB")
        
        # Taille adaptée pour le traitement initial
        parsing_size = (256, 352) if optimise_performance else (384, 512)
        model_parse, _ = self.parsing(src_image.resize(parsing_size))
        keypoints = self.openpose(src_image.resize(parsing_size))
        
        mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
        mask = mask.resize((width, height))
        
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
        
        # Inférence avec paramètres adaptés au type de vêtement
        # Ajuster le nombre d'étapes en fonction du type de vêtement
        adjusted_steps = step
        if vt_garment_type == "dresses":
            # Les robes sont plus complexes et nécessitent plus d'étapes
            adjusted_steps = min(step + 5, 50)
        
        # Inférence
        output = self.vt_inference_hd(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=adjusted_steps,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)

# Initialisation du prédicteur
try:
    print("Création du prédicteur Leffa (mode essayage optimisé pour RTX 4060)...")
    leffa_predictor = LeffaEssayagePredictor()
    
    # Préparation des exemples
    example_dir = "./ckpts/examples"
    person1_images = list_dir(f"{example_dir}/person1")
    garment_images = list_dir(f"{example_dir}/garment")
    
    print("Création de l'interface Gradio...")
    
    # Interface Gradio améliorée
    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink)) as demo:
        gr.Markdown("# 🌟 Leffa Ultra - Essayage Virtuel Optimisé 🌟")
        gr.Markdown("Application optimisée pour l'essayage virtuel de vêtements sur GPU RTX 4060")
        
        with gr.Row():
            with gr.Column(scale=1):
                src_image = gr.Image(type="filepath", label="📷 Image de la personne")
                ref_image = gr.Image(type="filepath", label="👕 Image du vêtement")
                
                with gr.Group():
                    gr.Markdown("### Type de vêtement")
                    vt_garment_type = gr.Radio(
                        choices=[
                            "upper_body", 
                            "lower_body", 
                            "dresses"
                        ],
                        value="upper_body",
                        label="Sélectionnez le type de vêtement",
                        info="Haut du corps, bas du corps, ou robe complète"
                    )
                
                with gr.Accordion("⚙️ Paramètres de qualité", open=True):
                    quality_preset = gr.Radio(
                        choices=["Rapide", "Standard", "Haute qualité"],
                        value="Standard",
                        label="Préréglage de qualité",
                        info="Impact sur la vitesse et la qualité du rendu"
                    )
                    
                    optimise_performance = gr.Checkbox(
                        value=True,
                        label="Optimiser pour les performances",
                        info="Optimise la mémoire et la vitesse de traitement"
                    )
                
                with gr.Accordion("🔧 Paramètres avancés", open=False):
                    preprocess_garment = gr.Checkbox(
                        value=False,
                        label="Prétraiter l'image du vêtement (PNG transparent uniquement)",
                        info="Utile pour les vêtements avec fond transparent"
                    )
                    
                    ref_acceleration = gr.Checkbox(
                        value=True, 
                        label="Accélération référence",
                        info="Accélère le processus de génération"
                    )
                    
                    vt_repaint = gr.Checkbox(
                        value=False, 
                        label="Repeindre",
                        info="Améliore la qualité mais ralentit le processus"
                    )
                    
                    step = gr.Slider(
                        minimum=1, 
                        maximum=50, 
                        value=20, 
                        step=1, 
                        label="Nombre d'étapes d'inférence",
                        info="Plus d'étapes = meilleure qualité mais plus lent"
                    )
                    
                    scale = gr.Slider(
                        minimum=1, 
                        maximum=10, 
                        value=3.0, 
                        step=0.5, 
                        label="Échelle de guidage",
                        info="Contrôle l'importance de l'image de référence"
                    )
                    
                    seed = gr.Number(
                        value=42, 
                        label="Graine aléatoire",
                        info="Utiliser la même valeur pour des résultats reproductibles"
                    )
                
                predict_btn = gr.Button("✨ Essayer le vêtement", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                result_image = gr.Image(label="🖼️ Résultat")
                with gr.Accordion("Visualisations techniques", open=False):
                    mask_image = gr.Image(label="Masque")
                    densepose_image = gr.Image(label="DensePose")
        
        # Logique pour ajuster les paramètres en fonction du préréglage de qualité
        def update_quality_params(preset):
            if preset == "Rapide":
                return 10, 2.0, True, False, True
            elif preset == "Haute qualité":
                return 35, 4.0, False, True, False
            else:  # Standard
                return 20, 3.0, True, False, True
        
        # Mettre à jour les paramètres lorsque le préréglage change
        quality_preset.change(
            fn=update_quality_params,
            inputs=[quality_preset],
            outputs=[step, scale, ref_acceleration, vt_repaint, optimise_performance]
        )
        
        # Fonction de prédiction mise à jour
        def predict_wrapper(src_image, ref_image, quality_preset, vt_garment_type, preprocess_garment, 
                           ref_acceleration, vt_repaint, step, scale, seed, optimise_performance):
            # Obtenir les paramètres du préréglage
            params = update_quality_params(quality_preset)
            
            # Utiliser les paramètres du préréglage à moins qu'ils n'aient été modifiés manuellement
            if step == params[0]:
                step = params[0]
            if scale == params[1]:
                scale = params[1]
            if ref_acceleration == params[2]:
                ref_acceleration = params[2]
            if vt_repaint == params[3]:
                vt_repaint = params[3]
            if optimise_performance == params[4]:
                optimise_performance = params[4]
            
            # Appeler la fonction de prédiction
            gen_image, mask, densepose = leffa_predictor.leffa_predict_vt(
                src_image, ref_image, ref_acceleration, step, scale, seed, 
                "viton_hd", vt_garment_type, vt_repaint, preprocess_garment, optimise_performance
            )
            
            return gen_image, mask, densepose
        
        # Connecter les entrées et sorties
        predict_btn.click(
            fn=predict_wrapper,
            inputs=[
                src_image, ref_image, quality_preset, vt_garment_type, preprocess_garment,
                ref_acceleration, vt_repaint, step, scale, seed, optimise_performance
            ],
            outputs=[result_image, mask_image, densepose_image]
        )
        
        # Exemples améliorés
        gr.Examples(
            examples=[
                [person1_images[0], garment_images[0], "Standard", "upper_body", False, True, False, 20, 3.0, 42, True],
                [person1_images[1], garment_images[1], "Haute qualité", "dresses", False, False, True, 35, 4.0, 42, False],
            ],
            inputs=[
                src_image, ref_image, quality_preset, vt_garment_type, preprocess_garment,
                ref_acceleration, vt_repaint, step, scale, seed, optimise_performance
            ],
            outputs=[result_image, mask_image, densepose_image],
        )
    
    print("\n===== INTERFACE LEFFA PRÊTE (MODE ESSAYAGE UNIQUEMENT AVEC GPU) =====")
    demo.launch(server_name="127.0.0.1", server_port=7860)

except Exception as e:
    import traceback
    print("Une erreur est survenue lors de l'initialisation:")
    print(traceback.format_exc())
    print(e)
