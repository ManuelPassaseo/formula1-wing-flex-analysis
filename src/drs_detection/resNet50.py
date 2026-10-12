import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import cv2
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configurazione
TARGET_SIZE = (224, 224)
BATCH_SIZE = 8
BASE_DIR = '/work/cvcs2025/passaseo_pivetti_vaccari/drs_finale/drs_dataset'
TEST_DIR = '/work/cvcs2025/passaseo_pivetti_vaccari/drs_finale/drs_dataset/test'
MANUAL_ANNOTATIONS_FILE = '/work/cvcs2025/passaseo_pivetti_vaccari/drs_finale/allvideos_annotation.csv'

# Directory di output
OUTPUT_DIR = '/work/cvcs2025/passaseo_pivetti_vaccari/drs_finale/resNet50/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        pooling='avg'
    )
    
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def train_model(epochs):
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_dir = os.path.join(BASE_DIR, 'train')
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        subset='validation'
    )
    
    # Modello path con epoche incluse
    model_path = f'/work/cvcs2025/passaseo_pivetti_vaccari/drs_finale/resNet50/drs_detector_resnet_epochs_{epochs}.keras'
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    model = build_model()
    
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )
    
    return model, history.history, model_path, train_generator, validation_generator

def plot_training_history(history, epochs, output_dir):
    """Plot training history for a single model"""
    
    # Estrai le metriche dalla storia
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    precision = history['precision']
    val_precision = history['val_precision']
    recall = history['recall']
    val_recall = history['val_recall']
    
    # Crea i subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Grafico accuracy
    axes[0, 0].plot(acc, label='Training Accuracy')
    axes[0, 0].plot(val_acc, label='Validation Accuracy')
    axes[0, 0].set_title(f'Training and Validation Accuracy (Epochs: {epochs})')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Grafico loss
    axes[0, 1].plot(loss, label='Training Loss')
    axes[0, 1].plot(val_loss, label='Validation Loss')
    axes[0, 1].set_title(f'Training and Validation Loss (Epochs: {epochs})')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Grafico precision
    axes[1, 0].plot(precision, label='Training Precision')
    axes[1, 0].plot(val_precision, label='Validation Precision')
    axes[1, 0].set_title(f'Training and Validation Precision (Epochs: {epochs})')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Grafico recall
    axes[1, 1].plot(recall, label='Training Recall')
    axes[1, 1].plot(val_recall, label='Validation Recall')
    axes[1, 1].set_title(f'Training and Validation Recall (Epochs: {epochs})')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_history_epochs_{epochs}.png'))
    plt.close()

def find_best_threshold_roc(model, generator, epochs):
    print("\nRicerca della miglior threshold usando la ROC curve...")
    
    # Predizioni
    y_pred = model.predict(generator, verbose=1)
    y_true = generator.classes
    
    # Calcola la ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Trova il punto sulla ROC curve più vicino all'angolo in alto a sinistra (0,1)
    dist_to_ideal = np.sqrt((1 - tpr)**2 + fpr**2)
    best_idx = np.argmin(dist_to_ideal)
    best_threshold = thresholds[best_idx]
    
    print(f"Miglior threshold: {best_threshold:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"FPR: {fpr[best_idx]:.4f}, TPR: {tpr[best_idx]:.4f}")
    
    # Plot della ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, 
                label=f'Best threshold: {best_threshold:.4f}\nFPR: {fpr[best_idx]:.4f}, TPR: {tpr[best_idx]:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curve_epochs_{epochs}.png'))
    plt.close()
    
    return best_threshold

def evaluate_dataset(model, generator, dataset_name, threshold, epochs):
    print(f"\nValutazione su {dataset_name} con threshold {threshold}...")
    
    # Predizioni
    y_pred = model.predict(generator, verbose=1)
    y_pred_classes = (y_pred > threshold).astype(int).flatten()
    y_true = generator.classes
    
    # Metriche
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, zero_division=0)
    recall = recall_score(y_true, y_pred_classes, zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Report di classificazione
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=['DRS OFF', 'DRS ON'], zero_division=0))
    
    # Matrice di confusione
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['DRS OFF', 'DRS ON'],
                yticklabels=['DRS OFF', 'DRS ON'])
    plt.title(f'Confusion Matrix - {dataset_name} (Epochs: {epochs}, Threshold: {threshold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    output_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_confusion_matrix_epochs_{epochs}.png')
    plt.savefig(output_path)
    plt.close()
    
    return {
        'dataset': dataset_name,
        'epochs': epochs,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'total_samples': len(y_true),
        'drs_on_predicted': sum(y_pred_classes),
        'drs_off_predicted': len(y_pred_classes) - sum(y_pred_classes)
    }

def evaluate_test_set_with_manual_annotations(model, test_dir, annotations_file, threshold, epochs):
    print(f"\nValutazione del Test Set con annotazioni manuali (threshold: {threshold})...")
    
    # Carica le annotazioni manuali
    if not os.path.exists(annotations_file):
        print(f"File di annotazioni manuali non trovato: {annotations_file}")
        return None
    
    manual_annotations = pd.read_csv(annotations_file)
    
    # Prepara il generatore per il test set usando DataFrame
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input
    )
    
    # Usa flow_from_dataframe invece di flow_from_directory
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=manual_annotations,
        directory=test_dir,
        x_col='frame',
        y_col='manual_label',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    # Ottieni le predizioni
    test_pred = model.predict(test_generator, verbose=1)
    test_pred_classes = (test_pred > threshold).astype(int).flatten()
    
    # Ottieni le etichette vere dal generatore
    y_true = test_generator.classes
    
    # Calcola le metriche
    accuracy = accuracy_score(y_true, test_pred_classes)
    precision = precision_score(y_true, test_pred_classes, zero_division=0)
    recall = recall_score(y_true, test_pred_classes, zero_division=0)
    f1 = f1_score(y_true, test_pred_classes, zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Report di classificazione
    print("\nClassification Report:")
    print(classification_report(y_true, test_pred_classes, target_names=['DRS OFF', 'DRS ON'], zero_division=0))
    
    # Matrice di confusione
    cm = confusion_matrix(y_true, test_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['DRS OFF', 'DRS ON'],
                yticklabels=['DRS OFF', 'DRS ON'])
    plt.title(f'Confusion Matrix - Test Set (Epochs: {epochs}, Threshold: {threshold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    output_path = os.path.join(OUTPUT_DIR, f'Test_Set_confusion_matrix_epochs_{epochs}.png')
    plt.savefig(output_path)
    plt.close()
    
    # Prepara i risultati dettagliati
    filenames = test_generator.filenames
    frame_names = [os.path.basename(f) for f in filenames]
    predictions_df = pd.DataFrame({
        'frame': frame_names,
        'prediction': test_pred.flatten(),
        'predicted_label': ['DRS ON' if p > threshold else 'DRS OFF' for p in test_pred],
        'true_label': ['DRS ON' if t == 1 else 'DRS OFF' for t in y_true]
    })
    
    # Salva i risultati dettagliati
    output_path = os.path.join(OUTPUT_DIR, f'test_set_detailed_evaluation_epochs_{epochs}.csv')
    predictions_df.to_csv(output_path, index=False)
    print(f"Risultati dettagliati salvati in '{output_path}'")
    
    return {
        'dataset': 'Test Set',
        'epochs': epochs,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'total_samples': len(y_true),
        'drs_on_predicted': sum(test_pred_classes),
        'drs_off_predicted': len(test_pred_classes) - sum(test_pred_classes)
    }

def create_detection_video(model, frame_dir, output_path, threshold, fps=30):
    """Crea video con rilevamento DRS usando la threshold ottimale"""

    if not os.path.isdir(frame_dir):
        raise ValueError(f"La directory '{frame_dir}' non esiste")
    
    valid_ext = ('.png', '.jpg', '.jpeg')
    frames = [f for f in os.listdir(frame_dir) 
              if f.lower().endswith(valid_ext)]
    
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"Nessun frame trovato in '{frame_dir}'")
    
    # Ordinamento naturale dei frame
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    
    frames.sort(key=natural_sort_key)
    
    # Verifica video
    first_frame_path = os.path.join(frame_dir, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"Impossibile leggere il frame '{frames[0]}'")
    
    h, w = first_frame.shape[:2]
    
    # Inizializzazione video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not video.isOpened():
        raise RuntimeError("Impossibile creare le video writer")
    
    print(f"\nInizio elaborazione di {total_frames} frame...")
    print(f"Dimensione frame: {w}x{h} | FPS: {fps}")
    print(f"Threshold utilizzata: {threshold:.4f}")
    
    processed_count = 0
    skipped_count = 0
    drs_on_count = 0
    drs_off_count = 0
    
    try:
        for i, frame_file in enumerate(frames):
            frame_path = os.path.join(frame_dir, frame_file)
            original_frame = cv2.imread(frame_path)
            
            if original_frame is None:
                print(f"Attenzione: Frame '{frame_file}' non valido, verrà saltato")
                skipped_count += 1
                continue
                
            # Manteniamo una copia ridimensionata per la predizione
            resized_frame = cv2.resize(original_frame.copy(), TARGET_SIZE)
            
            # Preprocessing e predizione
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            processed_frame = tf.keras.applications.resnet.preprocess_input(rgb_frame)
            prediction = model.predict(np.array([processed_frame]), verbose=0)[0][0]
            
            # Utilizza la threshold ottimale
            drs_status = "DRS ON" if prediction > threshold else "DRS OFF"
            
            if drs_status == "DRS ON":
                drs_on_count += 1
            else:
                drs_off_count += 1
            
            text = f"{drs_status}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            thickness = 6
            
            color = (0, 255, 0) if drs_status == "DRS ON" else (0, 0, 255)
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = text_size[1] + 50
            
            cv2.putText(
                original_frame, 
                text, 
                (text_x, text_y), 
                font, 
                font_scale, 
                color, 
                thickness, 
                cv2.LINE_AA
            )
            
            # Scrittura frame nel video
            video.write(original_frame)
            processed_count += 1
            
    except Exception as e:
        print(f"Errore durante l'elaborazione del frame {frame_file}: {str(e)}")
        skipped_count += 1
    finally:
        video.release()
        
        # Statistiche finali
        print("\n" + "="*50)
        print(f"ELABORAZIONE COMPLETATA")
        print(f"Frame totali: {total_frames}")
        print(f"Frame elaborati: {processed_count}")
        print(f"Frame saltati: {skipped_count}")
        print(f"DRS ON: {drs_on_count} frame ({drs_on_count/processed_count*100:.1f}%)")
        print(f"DRS OFF: {drs_off_count} frame ({drs_off_count/processed_count*100:.1f}%)")
        print(f"Video creato: {output_path}")
        print(f"Dimensioni: {w}x{h} | FPS: {fps}")
        print("="*50)

def plot_training_metrics_comparison(all_histories, test_results, epochs_list, output_dir):
    """Plot training, validation and test metrics across different epoch configurations"""
    
    # Prepare data
    train_losses = [h['loss'][-1] for h in all_histories]
    val_losses = [h['val_loss'][-1] for h in all_histories]
    train_accs = [h['accuracy'][-1] for h in all_histories]
    val_accs = [h['val_accuracy'][-1] for h in all_histories]
    train_precisions = [h['precision'][-1] for h in all_histories]
    val_precisions = [h['val_precision'][-1] for h in all_histories]
    train_recalls = [h['recall'][-1] for h in all_histories]
    val_recalls = [h['val_recall'][-1] for h in all_histories]
    
    # Extract test metrics
    test_accs = [r['accuracy'] for r in test_results if r is not None]
    test_precisions = [r['precision'] for r in test_results if r is not None]
    test_recalls = [r['recall'] for r in test_results if r is not None]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss plot
    axes[0, 0].plot(epochs_list, train_losses, 'b-', label='Training Loss', marker='o')
    axes[0, 0].plot(epochs_list, val_losses, 'r-', label='Validation Loss', marker='o')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Final Loss vs Number of Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs_list, train_accs, 'b-', label='Training Accuracy', marker='o')
    axes[0, 1].plot(epochs_list, val_accs, 'r-', label='Validation Accuracy', marker='o')
    axes[0, 1].plot(epochs_list, test_accs, 'g-', label='Test Accuracy', marker='o')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Final Accuracy vs Number of Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision plot
    axes[1, 0].plot(epochs_list, train_precisions, 'b-', label='Training Precision', marker='o')
    axes[1, 0].plot(epochs_list, val_precisions, 'r-', label='Validation Precision', marker='o')
    axes[1, 0].plot(epochs_list, test_precisions, 'g-', label='Test Precision', marker='o')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Final Precision vs Number of Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall plot
    axes[1, 1].plot(epochs_list, train_recalls, 'b-', label='Training Recall', marker='o')
    axes[1, 1].plot(epochs_list, val_recalls, 'r-', label='Validation Recall', marker='o')
    axes[1, 1].plot(epochs_list, test_recalls, 'g-', label='Test Recall', marker='o')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Final Recall vs Number of Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics_comparison.png'))
    plt.close()
    
    # Create a comprehensive metrics dataframe
    metrics_df = pd.DataFrame({
        'epochs': epochs_list,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accs,
        'val_accuracy': val_accs,
        'test_accuracy': test_accs,
        'train_precision': train_precisions,
        'val_precision': val_precisions,
        'test_precision': test_precisions,
        'train_recall': train_recalls,
        'val_recall': val_recalls,
        'test_recall': test_recalls
    })
    
    metrics_df.to_csv(os.path.join(output_dir, 'training_metrics_details.csv'), index=False)
    print(f"Dettagli delle metriche salvati in '{os.path.join(output_dir, 'training_metrics_details.csv')}'")

if __name__ == "__main__":
    # Lista di epoche da testare
    epochs_list = [10, 20, 30, 40, 50]
    results = []
    all_histories = []  # Per salvare le storie di tutti i training
    test_results_list = []  # Per salvare i risultati dei test
    
    for epochs in epochs_list:
        print(f"\n{'='*60}")
        print(f"ALLENAMENTO E VALUTAZIONE MODELLO CON {epochs} EPOCHE")
        print(f"{'='*60}")
        
        # Allenamento del modello
        model, history, model_path, train_generator, val_generator = train_model(epochs)
        all_histories.append(history)  # Salva la storia
        
        # Plot della storia di addestramento per questo modello
        plot_training_history(history, epochs, OUTPUT_DIR)
        
        # Trova la miglior threshold usando la ROC curve sul validation set
        best_threshold = find_best_threshold_roc(model, val_generator, epochs)
        
        # Valutazione su training set
        train_results = evaluate_dataset(model, train_generator, 'Training Set', best_threshold, epochs)
        
        # Valutazione su validation set
        val_results = evaluate_dataset(model, val_generator, 'Validation Set', best_threshold, epochs)
        
        # Valutazione del test set con annotazioni manuali
        test_results = evaluate_test_set_with_manual_annotations(
            model, TEST_DIR, MANUAL_ANNOTATIONS_FILE, best_threshold, epochs
        )
        test_results_list.append(test_results)  # Salva i risultati del test
        
        # Aggiungi tutti i risultati alla lista
        if train_results:
            results.append(train_results)
        if val_results:
            results.append(val_results)
        if test_results:
            results.append(test_results)
        
        # Salva il modello
        model.save(model_path)
        print(f"Modello salvato in: {model_path}")
    
    # Plot dei grafici di confronto finale
    plot_training_metrics_comparison(all_histories, test_results_list, epochs_list, OUTPUT_DIR)
    
    # Salva i risultati complessivi
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'model_evaluation_results_all_epochs.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nRisultati complessivi salvati in '{output_path}'")
    
    # Trova il modello migliore in base all'F1-score sul test set
    validation_results_df = results_df[results_df['dataset'] == 'Validation Set']
    if not validation_results_df.empty:
        best_result = validation_results_df.loc[validation_results_df['recall'].idxmax()]
        print(f"\nMiglior modello: {best_result['epochs']} epoche")
        print(f"Validation Recall: {best_result['recall']:.4f}")
        print(f"Threshold ottimale: {best_result['threshold']:.4f}")
        
        # Carica il modello migliore
        best_model_path = f'/work/cvcs2025/passaseo_pivetti_vaccari/drs_finale/resNet50/drs_detector_resnet_epochs_{best_result["epochs"]}.keras'
        best_model = tf.keras.models.load_model(best_model_path)
        
        # Crea video con il modello migliore
        output_video = f"/work/cvcs2025/passaseo_pivetti_vaccari/drs_finale/resNet50/prediction_video_best_model_epochs_{best_result['epochs']}.mp4"
        
        create_detection_video(
            model=best_model,
            frame_dir=TEST_DIR,
            output_path=output_video,
            threshold=best_result['threshold'],
            fps=30
        )
    else:
        print("\nNessun risultato del test set trovato per determinare il modello migliore.")