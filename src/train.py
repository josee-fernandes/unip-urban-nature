import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
import random

# Suprimir warnings do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = todos, 1 = info, 2 = warnings, 3 = errors
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# Configurar reprodutibilidade
def set_seed(seed=42):
    """Define seeds para garantir reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_simple_cnn(input_shape=(128,128,3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32,3,activation='relu',padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu',padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu',padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_transfer_mobilenet(input_shape=(128,128,3)):
    base = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet', pooling='avg')
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history, outdir):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(outdir, 'train_history.png'))
    plt.close()

def plot_confusion_matrix(cm, classes, outpath):
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(args):
    # Configurar reprodutibilidade
    set_seed(args.seed)
    
    outdir = args.output
    os.makedirs(outdir, exist_ok=True)
    img_size = (args.height, args.width)
    batch_size = args.batch_size
    
    # Verificar se existe pasta test separada
    test_dir = os.path.join(os.path.dirname(args.data_dir), 'test')
    has_separate_test = os.path.exists(test_dir) and os.path.isdir(test_dir)
    
    # Load datasets
    if has_separate_test:
        # Se existe pasta test separada, usar train/val/test
        print("Usando estrutura train/val/test separada")
        train_ds = image_dataset_from_directory(
            args.data_dir,
            validation_split=args.validation_split,
            subset="training",
            seed=args.seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='binary'
        )
        val_ds = image_dataset_from_directory(
            args.data_dir,
            validation_split=args.validation_split,
            subset="validation",
            seed=args.seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='binary'
        )
        test_ds = image_dataset_from_directory(
            test_dir,
            subset=None,
            seed=args.seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='binary'
        )
    else:
        # Se não existe pasta test, dividir train/val apenas
        print("Usando divisão train/val (sem pasta test separada)")
        train_ds = image_dataset_from_directory(
            args.data_dir,
            validation_split=args.validation_split,
            subset="training",
            seed=args.seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='binary'
        )
        val_ds = image_dataset_from_directory(
            args.data_dir,
            validation_split=args.validation_split,
            subset="validation",
            seed=args.seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='binary'
        )
        test_ds = None
    
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    input_shape = (img_size[0], img_size[1], 3)
    if args.model == 'simple':
        model = build_simple_cnn(input_shape)
    else:
        model = build_transfer_mobilenet(input_shape)

    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(outdir, 'best_model.h5'), save_best_only=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb)
    plot_history(history, outdir)

    # Evaluate on test dataset (se disponível) ou validation dataset
    if test_ds is not None:
        print("Avaliando no conjunto de teste...")
        eval_ds = test_ds
        eval_name = "test"
    else:
        print("Avaliando no conjunto de validação (test não disponível)...")
        eval_ds = val_ds
        eval_name = "val"
    
    eval_images = []
    eval_labels = []
    for batch_images, batch_labels in eval_ds:
        eval_images.append(batch_images.numpy())
        # Garantir que labels sejam 1D
        labels_flat = batch_labels.numpy().flatten()
        eval_labels.append(labels_flat)
    X_eval = np.vstack(eval_images)
    y_eval = np.concatenate(eval_labels, axis=0)
    preds_prob = model.predict(X_eval, verbose=1).ravel()
    preds = (preds_prob >= 0.5).astype(int)

    # metrics
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    cm = confusion_matrix(y_eval, preds)
    plot_confusion_matrix(cm, class_names, os.path.join(outdir, f'confusion_matrix_{eval_name}.png'))
    prfs = precision_recall_fscore_support(y_eval, preds, labels=[0,1])
    df = pd.DataFrame({
        'class': class_names,
        'precision': prfs[0],
        'recall': prfs[1],
        'f1': prfs[2],
        'support': prfs[3]
    })
    df.to_csv(os.path.join(outdir, f'classification_report_{eval_name}.csv'), index=False)

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_eval, preds_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic ({eval_name})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outdir, f'roc_curve_{eval_name}.png'))
    plt.close()

    print(f"\n{'='*50}")
    print(f"Resultados no conjunto de {eval_name}:")
    print(f"{'='*50}")
    print("Saved outputs to:", outdir)
    print(df)
    print(f"AUC: {roc_auc:.4f}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina modelo de classificação binária (Urban vs Nature)')
    parser.add_argument('--data_dir', type=str, default='data/raw/train', 
                        help='Caminho para pasta train com subpastas de classes')
    parser.add_argument('--output', type=str, default='outputs', 
                        help='Diretório de saída para modelos e resultados')
    parser.add_argument('--model', type=str, default='transfer', choices=['transfer','simple'],
                        help='Tipo de modelo: transfer (MobileNetV2) ou simple (CNN simples)')
    parser.add_argument('--height', type=int, default=128, help='Altura das imagens')
    parser.add_argument('--width', type=int, default=128, help='Largura das imagens')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas')
    parser.add_argument('--validation_split', type=float, default=0.2, 
                        help='Proporção para validação (usado apenas se não houver pasta test separada)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed para reprodutibilidade (random_state)')
    args = parser.parse_args()
    main(args)
