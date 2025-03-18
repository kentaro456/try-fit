# Leffa Ultra - Essayage Virtuel Optimisé pour RTX 4060

## À propos du projet

Leffa Ultra est une version optimisée de [Leffa](https://github.com/franciszzj/Leffa), un framework d'essayage virtuel et de transfert de pose basé sur l'intelligence artificielle. Cette version a été spécialement configurée pour fonctionner de manière optimale sur les cartes graphiques NVIDIA RTX 4060, offrant un équilibre parfait entre performances et qualité d'image.

![Leffa Demo](https://huggingface.co/franciszzj/Leffa/resolve/main/assets/teaser.png)

## Caractéristiques principales

- **Optimisation RTX 4060** : Configurations spécifiques pour maximiser les performances sur les GPU RTX 4060
- **Interface utilisateur améliorée** : Interface Gradio intuitive avec paramètres avancés
- **Préréglages de qualité** : Options Rapide, Standard et Haute Qualité pour adapter les performances à vos besoins
- **Support multi-vêtements** : Prise en charge des hauts, bas et robes complètes
- **Optimisation automatique** : Ajustement des paramètres en fonction du type de vêtement pour des résultats optimaux

## Configuration matérielle recommandée

- **GPU** : NVIDIA RTX 4060 (ou supérieur)
- **RAM** : 16 Go minimum
- **Stockage** : 10 Go d'espace disque libre pour les modèles et l'application
- **CPU** : AMD Ryzen 7 7844 ou équivalent Intel

## Installation

1. Créez un environnement Python (version 3.10 recommandée)
```bash
conda create -n leffa python=3.10
conda activate leffa
```

2. Installez les dépendances
```bash
pip install -r requirements.txt
```

3. Lancez l'application 
```bash
python run_leffa_gpu_fix.py
```

## Guide d'utilisation

1. **Sélection du type de vêtement** : Choisissez entre "Haut", "Bas" ou "Robe"
2. **Qualité de l'essayage** : Sélectionnez le préréglage qui correspond à vos besoins
   - **Rapide** : Résultats plus rapides, qualité légèrement réduite
   - **Standard** : Bon équilibre entre vitesse et qualité
   - **Haute qualité** : Résultats optimaux, mais plus lent

3. **Options avancées** :
   - **Étapes d'inférence** : Augmenter pour plus de détails, mais le processus sera plus lent
   - **Échelle de guidage** : Contrôle l'importance de l'image de référence
   - **Accélération référence** : Active l'accélération du traitement des références
   - **Repeindre** : Améliore la qualité de l'image finale au prix de performances réduites
   - **Optimisation des performances** : Ajuste automatiquement la résolution de l'image selon le vêtement

## Optimisations techniques

- **CUDA Benchmark** : `torch.backends.cudnn.benchmark = True` pour accélérer les opérations convolutives
- **Vidage cache CUDA** : Nettoyage du cache GPU au démarrage pour libérer la mémoire
- **Opérations CPU/GPU hybrides** : Certaines fonctions (NMS, ROI Align) sont exécutées sur CPU puis renvoyées sur GPU
- **Précision réduite** : Utilisation de la précision float16 pour les modèles afin d'économiser la mémoire
- **Allocation mémoire optimisée** : Configuration `PYTORCH_CUDA_ALLOC_CONF` pour éviter la fragmentation mémoire

## Résolution de problèmes

- **Erreur de mémoire CUDA** : Essayez de fermer d'autres applications utilisant le GPU ou sélectionnez le préréglage "Rapide"
- **Lenteur** : Vérifiez que les pilotes NVIDIA sont à jour (version 560 ou supérieure recommandée)
- **Artefacts visuels** : Essayez l'option "Repeindre" dans les paramètres avancés

## Remerciements

Ce projet est basé sur [Leffa](https://github.com/franciszzj/Leffa) par Zhou et al. Les optimisations GPU et l'interface améliorée ont été développées spécifiquement pour les cartes RTX 4060.

Citation originale :
```
@article{zhou2024learning,
  title={Learning Flow Fields in Attention for Controllable Person Image Generation}, 
  author={Zhou, Zijian and Liu, Shikun and Han, Xiao and Liu, Haozhe and Ng, Kam Woh and Xie, Tian and Cong, Yuren and Li, Hang and Xu, Mengmeng and Pérez-Rúa, Juan-Manuel and Patel, Aditya and Xiang, Tao and Shi, Miaojing and He, Sen},
  journal={arXiv preprint arXiv:2412.08486},
  year={2024},
}