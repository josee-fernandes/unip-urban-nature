#!/usr/bin/env python3
"""
Script para baixar e organizar datasets do Kaggle para o projeto Urban vs Nature.

Uso:
    python scripts/download_kaggle_dataset.py --dataset urban --max_images 500
    python scripts/download_kaggle_dataset.py --dataset nature --max_images 500
"""

import os
import argparse
import shutil
from pathlib import Path
import random

def organize_kaggle_dataset(dataset_path, output_dir, class_name, max_images=None, seed=42):
    """
    Organiza imagens de um dataset do Kaggle na estrutura esperada pelo projeto.
    
    Args:
        dataset_path: Caminho para a pasta baixada do Kaggle
        output_dir: Diretório de saída (data/raw/train/ ou data/raw/test/)
        class_name: Nome da classe ('urban' ou 'nature')
        max_images: Número máximo de imagens para copiar (None = todas)
        seed: Seed para seleção aleatória de imagens
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    if not dataset_path.exists():
        raise ValueError(f"Pasta não encontrada: {dataset_path}")
    
    # Criar diretório de saída
    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar todas as imagens (recursivamente)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(dataset_path.rglob(f'*{ext}'))
        all_images.extend(dataset_path.rglob(f'*{ext.upper()}'))
    
    if not all_images:
        raise ValueError(f"Nenhuma imagem encontrada em {dataset_path}")
    
    print(f"Encontradas {len(all_images)} imagens em {dataset_path}")
    
    # Selecionar amostra se necessário
    if max_images and len(all_images) > max_images:
        random.seed(seed)
        selected_images = random.sample(all_images, max_images)
        print(f"Selecionando {max_images} imagens aleatoriamente...")
    else:
        selected_images = all_images
    
    # Copiar imagens
    copied = 0
    for i, img_path in enumerate(selected_images, 1):
        # Gerar nome único
        new_name = f"{class_name}_{i:05d}{img_path.suffix}"
        dest_path = class_dir / new_name
        
        try:
            shutil.copy2(img_path, dest_path)
            copied += 1
            if i % 50 == 0:
                print(f"  Copiadas {i}/{len(selected_images)} imagens...")
        except Exception as e:
            print(f"  Erro ao copiar {img_path}: {e}")
    
    print(f"\n✓ {copied} imagens copiadas para {class_dir}")
    return copied

def main():
    parser = argparse.ArgumentParser(
        description='Organiza datasets do Kaggle na estrutura do projeto'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['urban', 'nature'],
        help='Tipo de dataset: urban ou nature'
    )
    parser.add_argument(
        '--kaggle_path',
        type=str,
        required=True,
        help='Caminho para a pasta baixada do Kaggle'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/train',
        help='Diretório de saída (data/raw/train ou data/raw/test)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Número máximo de imagens para copiar (None = todas)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para seleção aleatória (se max_images for usado)'
    )
    
    args = parser.parse_args()
    
    # Mapear dataset para nome da classe
    class_name = args.dataset
    
    print(f"\n{'='*60}")
    print(f"Organizando dataset: {args.dataset}")
    print(f"Origem: {args.kaggle_path}")
    print(f"Destino: {args.output}/{class_name}")
    if args.max_images:
        print(f"Limite: {args.max_images} imagens")
    print(f"{'='*60}\n")
    
    try:
        organize_kaggle_dataset(
            args.kaggle_path,
            args.output,
            class_name,
            args.max_images,
            args.seed
        )
        print("\n✓ Organização concluída com sucesso!")
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

