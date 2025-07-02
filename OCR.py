import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Configuracion:
    TAMAÑO_IMG = 64
    TAMAÑO_PARCHE = 8
    NUM_PARCHES = (TAMAÑO_IMG // TAMAÑO_PARCHE) ** 2
    DIM_EMBEDDING = 128
    NUM_CABEZAS = 4
    NUM_CAPAS = 4
    DIM_MLP = 256
    ABANDONO = 0.1
    TAMAÑO_LOTE = 32
    TASA_APRENDIZAJE = 1e-4
    NUM_EPOCAS = 20
    DIVISION_VALIDACION = 0.2
    CLASES_OBJETIVO = 100  

configuracion = Configuracion()

class CargadorDatasetChino:
    def __init__(self, directorio_datos: str = "./datos_chinos"):
        self.directorio_datos = directorio_datos
        self.imagenes = []
        self.etiquetas = []
        self.codificador_etiquetas = LabelEncoder()
        
        os.makedirs(directorio_datos, exist_ok=True)
    
    def descargar_dataset_muestra(self):
        print("Buscando datos locales...")        
        extensiones_imagen = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        imagenes_encontradas = []
        
        for raiz, directorios, archivos in os.walk(self.directorio_datos):
            for archivo in archivos:
                if any(archivo.lower().endswith(ext) for ext in extensiones_imagen):
                    imagenes_encontradas.append(os.path.join(raiz, archivo))
        
        if imagenes_encontradas:
            print(f"Encontradas {len(imagenes_encontradas)} imágenes locales.")
            return self._cargar_imagenes_locales(imagenes_encontradas)
        else:
            print("No se encontraron imágenes locales. Generando dataset sintético...")
            return self._crear_dataset_sintetico()
    
    def _cargar_imagenes_locales(self, rutas_imagenes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        imagenes = []
        etiquetas = []
        
        for ruta_img in sorted(rutas_imagenes):
            try:
                img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue                
                img = cv2.resize(img, (configuracion.TAMAÑO_IMG, configuracion.TAMAÑO_IMG))
                img = img.astype(np.float32) / 255.0                
                nombre_archivo = os.path.basename(ruta_img)
                etiqueta = nombre_archivo.split('_')[0] if '_' in nombre_archivo else nombre_archivo.split('.')[0]
                imagenes.append(img)
                etiquetas.append(etiqueta)
                
            except Exception as e:
                print(f"Error cargando {ruta_img}: {e}")
                continue
        
        if not imagenes:
            return self._crear_dataset_sintetico()
        return np.array(imagenes), np.array(etiquetas)
    
    def _crear_dataset_sintetico(self) -> Tuple[np.ndarray, np.ndarray]:
        caracteres_chinos = [
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            '人', '大', '小', '中', '国', '文', '字', '学', '生', '好',
            '你', '我', '他', '她', '们', '的', '在', '是', '有', '这',
            '那', '了', '不', '就', '都', '和', '要', '到', '说', '时',
            '上', '下', '来', '去', '出', '可', '以', '会', '对', '自',
            '家', '水', '火', '土', '木', '金', '日', '月', '山', '手',
            '口', '门', '心', '马', '车', '鸟', '鱼', '羊', '牛', '虎',
            '龙', '蛇', '兔', '猴', '鸡', '狗', '猪', '鼠', '花', '树',
            '草', '叶', '风', '雨', '雪', '云', '天', '地', '海', '河',
            '山', '石', '沙', '土', '泥', '光', '电', '雷', '霜', '露'
        ]
        
        caracteres_chinos = caracteres_chinos[:configuracion.CLASES_OBJETIVO]
        imagenes = []
        etiquetas = []
        muestras_por_clase = 50  
        
        for caracter in caracteres_chinos:
            for _ in range(muestras_por_clase):
                # Generar imagen sintética del carácter
                img = self._generar_imagen_caracter(caracter)
                imagenes.append(img)
                etiquetas.append(caracter)
        
        return np.array(imagenes), np.array(etiquetas)
    
    def _generar_imagen_caracter(self, caracter: str) -> np.ndarray:
        img = np.ones((configuracion.TAMAÑO_IMG, configuracion.TAMAÑO_IMG), dtype=np.uint8) * 255        
        img_pil = Image.fromarray(img)
        
        try:
            from PIL import ImageDraw, ImageFont
            dibujo = ImageDraw.Draw(img_pil)
            
            try:
                rutas_fuentes = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/Hiragino Sans GB.ttc", 
                    "C:/Windows/Fonts/simsun.ttc",
                    "C:/Windows/Fonts/msyh.ttc",
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
                ]
                
                fuente = None
                for ruta_fuente in rutas_fuentes:
                    if os.path.exists(ruta_fuente):
                        fuente = ImageFont.truetype(ruta_fuente, 32)
                        break
                
                if fuente is None:
                    fuente = ImageFont.load_default()
                
            except:
                fuente = ImageFont.load_default()
            
            x = np.random.randint(8, 24)
            y = np.random.randint(8, 24)            
            color = np.random.randint(0, 128)            
            dibujo.text((x, y), caracter, fill=color, font=fuente)
            
        except:
            img = np.random.randint(0, 255, (configuracion.TAMAÑO_IMG, configuracion.TAMAÑO_IMG), dtype=np.uint8)
            img_pil = Image.fromarray(img)        
        arreglo_img = np.array(img_pil).astype(np.float32) / 255.0        
        ruido = np.random.normal(0, 0.05, arreglo_img.shape)
        arreglo_img = np.clip(arreglo_img + ruido, 0, 1)
        
        return arreglo_img
    
    def cargar_datos(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        imagenes, etiquetas = self.descargar_dataset_muestra()
        print(f"Cargadas {len(imagenes)} imágenes con {len(set(etiquetas))} clases únicas")
        etiquetas_codificadas = self.codificador_etiquetas.fit_transform(etiquetas)        
        num_clases = len(self.codificador_etiquetas.classes_)
        etiquetas_categoricas = keras.utils.to_categorical(etiquetas_codificadas, num_clases)        
        imagenes = np.expand_dims(imagenes, axis=-1)        
        X_entreno, X_val, y_entreno, y_val = train_test_split(
            imagenes, etiquetas_categoricas, 
            test_size=configuracion.DIVISION_VALIDACION, 
            random_state=42,
            stratify=etiquetas_codificadas
        )
        
        print(f"Datos de entrenamiento: {X_entreno.shape}")
        print(f"Datos de validación: {X_val.shape}")
        print(f"Número de clases: {num_clases}")
        
        return X_entreno, X_val, y_entreno, y_val

@keras.utils.register_keras_serializable()
class ExtractorParches(layers.Layer):
    def __init__(self, tamaño_parche, **kwargs):
        super().__init__(**kwargs)
        self.tamaño_parche = tamaño_parche

    def call(self, imagenes):
        tamaño_lote = tf.shape(imagenes)[0]
        parches = tf.image.extract_patches(
            images=imagenes,
            sizes=[1, self.tamaño_parche, self.tamaño_parche, 1],
            strides=[1, self.tamaño_parche, self.tamaño_parche, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        dimensiones_parche = parches.shape[-1]
        parches = tf.reshape(parches, [tamaño_lote, -1, dimensiones_parche])
        return parches
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tamaño_parche": self.tamaño_parche,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable()
class CodificadorParches(layers.Layer):
    def __init__(self, num_parches, dimension_proyeccion, **kwargs):
        super().__init__(**kwargs)
        self.num_parches = num_parches
        self.dimension_proyeccion = dimension_proyeccion
        self.proyeccion = layers.Dense(units=dimension_proyeccion)
        self.embedding_posicion = layers.Embedding(
            input_dim=num_parches, output_dim=dimension_proyeccion
        )

    def call(self, parche):
        posiciones = tf.range(start=0, limit=self.num_parches, delta=1)
        codificado = self.proyeccion(parche) + self.embedding_posicion(posiciones)
        return codificado
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_parches": self.num_parches,
            "dimension_proyeccion": self.dimension_proyeccion,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def crear_vision_transformer(num_clases: int) -> keras.Model:
    entradas = layers.Input(shape=(configuracion.TAMAÑO_IMG, configuracion.TAMAÑO_IMG, 1))    
    parches = ExtractorParches(configuracion.TAMAÑO_PARCHE)(entradas)    
    parches_codificados = CodificadorParches(configuracion.NUM_PARCHES, configuracion.DIM_EMBEDDING)(parches)
    
    for _ in range(configuracion.NUM_CAPAS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(parches_codificados)        
        salida_atencion = layers.MultiHeadAttention(
            num_heads=configuracion.NUM_CABEZAS, 
            key_dim=configuracion.DIM_EMBEDDING // configuracion.NUM_CABEZAS,
            dropout=configuracion.ABANDONO
        )(x1, x1)
        
        x2 = layers.Add()([salida_atencion, parches_codificados])        
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)        
        x3 = layers.Dense(configuracion.DIM_MLP, activation="gelu")(x3)
        x3 = layers.Dropout(configuracion.ABANDONO)(x3)
        x3 = layers.Dense(configuracion.DIM_EMBEDDING)(x3)
        x3 = layers.Dropout(configuracion.ABANDONO)(x3)        
        parches_codificados = layers.Add()([x3, x2])
    
    representacion = layers.LayerNormalization(epsilon=1e-6)(parches_codificados)    
    representacion = layers.GlobalAveragePooling1D()(representacion)    
    representacion = layers.Dropout(configuracion.ABANDONO)(representacion)    
    salidas = layers.Dense(num_clases, activation="softmax")(representacion)
    modelo = keras.Model(inputs=entradas, outputs=salidas)
    return modelo

class EntrenadorOCRChino:
    def __init__(self):
        self.modelo = None
        self.cargador_dataset = CargadorDatasetChino()
        self.historial = None
        
    def preparar_datos(self):
        print("Preparando datos...")
        self.X_entreno, self.X_val, self.y_entreno, self.y_val = self.cargador_dataset.cargar_datos()
        self.num_clases = self.y_entreno.shape[1]
        
    def construir_modelo(self):
        print("Construyendo modelo...")
        self.modelo = crear_vision_transformer(self.num_clases)
        try:
            metricas = ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="precision_top_3")]
        except:
            print("Advertencia: TopKCategoricalAccuracy no disponible, usando solo precisión")
            metricas = ["accuracy"]
        
        self.modelo.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=configuracion.TASA_APRENDIZAJE,
                weight_decay=0.01
            ),
            loss="categorical_crossentropy",
            metrics=metricas
        )
        
        print("Resumen del modelo:")
        self.modelo.summary()
    
    def entrenar(self):
        if self.modelo is None:
            self.construir_modelo()
        
        print("Iniciando entrenamiento...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                "mejor_modelo_ocr_chino.h5",
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        self.historial = self.modelo.fit(
            self.X_entreno, self.y_entreno,
            batch_size=configuracion.TAMAÑO_LOTE,
            epochs=configuracion.NUM_EPOCAS,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("¡Entrenamiento completado!")
        
    def graficar_historial_entrenamiento(self):
        if self.historial is None:
            print("No hay historial de entrenamiento disponible.")
            return
        
        fig, ejes = plt.subplots(2, 2, figsize=(12, 8))
        
        ejes[0, 0].plot(self.historial.history['accuracy'], label='Precisión Entrenamiento')
        ejes[0, 0].plot(self.historial.history['val_accuracy'], label='Precisión Validación')
        ejes[0, 0].set_title('Precisión')
        ejes[0, 0].legend()        
        ejes[0, 1].plot(self.historial.history['loss'], label='Pérdida Entrenamiento')
        ejes[0, 1].plot(self.historial.history['val_loss'], label='Pérdida Validación')
        ejes[0, 1].set_title('Pérdida')
        ejes[0, 1].legend()
        
        if 'precision_top_3' in self.historial.history:
            ejes[1, 0].plot(self.historial.history['precision_top_3'], label='Top-3 Entreno')
            ejes[1, 0].plot(self.historial.history['val_precision_top_3'], label='Top-3 Validación')
            ejes[1, 0].set_title('Precisión Top-3')
            ejes[1, 0].legend()
        else:
            ejes[1, 0].text(0.5, 0.5, 'Precisión Top-3\nNo Disponible', 
                           ha='center', va='center', transform=ejes[1, 0].transAxes)
        
        if 'lr' in self.historial.history:
            ejes[1, 1].plot(self.historial.history['lr'])
            ejes[1, 1].set_title('Tasa de Aprendizaje')
            ejes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()

class PredictorOCRChino:
    def __init__(self, ruta_modelo: str = "mejor_modelo_ocr_chino.h5"):
        self.modelo = None
        self.codificador_etiquetas = None
        self.cargar_modelo(ruta_modelo)
    
    def cargar_modelo(self, ruta_modelo: str):
        if os.path.exists(ruta_modelo):
            print(f"Cargando modelo desde {ruta_modelo}")
            try:
                self.modelo = keras.models.load_model(ruta_modelo)
                print("¡Modelo cargado exitosamente!")                
                ruta_codificador = ruta_modelo.replace('.h5', '_codificador.pkl')
                if os.path.exists(ruta_codificador):
                    import pickle
                    with open(ruta_codificador, 'rb') as f:
                        self.codificador_etiquetas = pickle.load(f)
                        print("¡Codificador de etiquetas cargado exitosamente!")
                else:
                    print("Advertencia: No se encontró el codificador de etiquetas.")
                    
            except Exception as e:
                print(f"Error cargando el modelo: {e}")
                print("Posible solución: Reentrenar el modelo con esta versión del código.")
                
        else:
            print(f"No se encontró el modelo en {ruta_modelo}")
    
    def predecir_imagen(self, imagen, top_k: int = 5):
        if self.modelo is None:
            print("Modelo no cargado.")
            return None
        
        if isinstance(imagen, str):
            imagen = cv2.imread(imagen, cv2.IMREAD_GRAYSCALE)
        
        if imagen is None:
            print("Error: No se pudo cargar la imagen.")
            return None
        
        imagen = cv2.resize(imagen, (configuracion.TAMAÑO_IMG, configuracion.TAMAÑO_IMG))
        imagen = imagen.astype(np.float32) / 255.0
        imagen = np.expand_dims(imagen, axis=(0, -1))     
        predicciones = self.modelo.predict(imagen, verbose=0)        
        indices_top = np.argsort(predicciones[0])[::-1][:top_k]
        probabilidades_top = predicciones[0][indices_top]
        
        resultados = []
        for i, (idx, prob) in enumerate(zip(indices_top, probabilidades_top)):
            if self.codificador_etiquetas:
                caracter = self.codificador_etiquetas.inverse_transform([idx])[0]
            else:
                caracter = f"Clase_{idx}"
            resultados.append((caracter, prob))
        
        return resultados
    
    def predecir_lote(self, imagenes):
        if self.modelo is None:
            print("Modelo no cargado.")
            return None
        
        imagenes_procesadas = []
        for img in imagenes:
            if isinstance(img, str):
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img = cv2.resize(img, (configuracion.TAMAÑO_IMG, configuracion.TAMAÑO_IMG))
                img = img.astype(np.float32) / 255.0
                imagenes_procesadas.append(img)
        
        if not imagenes_procesadas:
            return None
        
        lote = np.array(imagenes_procesadas)
        lote = np.expand_dims(lote, axis=-1)
        predicciones = self.modelo.predict(lote, verbose=0)
        
        resultados = []
        for pred in predicciones:
            idx_top = np.argmax(pred)
            if self.codificador_etiquetas:
                caracter = self.codificador_etiquetas.inverse_transform([idx_top])[0]
            else:
                caracter = f"Clase_{idx_top}"
            resultados.append((caracter, pred[idx_top]))
        return resultados

def principal():
    print("=== OCR de Caracteres Chinos con TensorFlow (VERSIÓN CORREGIDA) ===")
    print("Este sistema usa Vision Transformer para reconocer caracteres chinos.")
    print("CORRECCIÓN: Ahora maneja correctamente las capas personalizadas.")
    print()
    
    entrenador = EntrenadorOCRChino()
    predictor = None  
    
    while True:
        print("\n--- Menú Principal ---")
        print("1. Preparar y entrenar modelo")
        print("2. Cargar modelo existente y predecir")
        print("3. Evaluar modelo en datos de validación")
        print("4. Mostrar historial de entrenamiento")
        print("5. Predecir imagen individual")
        print("6. Información del dataset")
        print("7. Salir")
        
        opcion = input("\nSelecciona una opción (1-7): ").strip()
        
        if opcion == '1':
            print("\n--- Entrenamiento del Modelo ---")
            entrenador.preparar_datos()
            entrenador.construir_modelo()
            entrenador.entrenar()
            entrenador.graficar_historial_entrenamiento()            
            import pickle
            with open('mejor_modelo_ocr_chino_codificador.pkl', 'wb') as f:
                pickle.dump(entrenador.cargador_dataset.codificador_etiquetas, f)
            print("Codificador de etiquetas guardado.")
            
        elif opcion == '2':
            ruta_modelo = input("Ruta del modelo (Enter para usar por defecto): ").strip()
            if not ruta_modelo:
                ruta_modelo = "mejor_modelo_ocr_chino.h5"
            
            predictor = PredictorOCRChino(ruta_modelo)
            
        elif opcion == '3':
            if entrenador.modelo and hasattr(entrenador, 'X_val'):
                print("\n--- Evaluación del Modelo ---")
                resultados = entrenador.modelo.evaluate(entrenador.X_val, entrenador.y_val, verbose=1)
                print(f"Pérdida: {resultados[0]:.4f}")
                print(f"Precisión: {resultados[1]:.4f}")
                if len(resultados) > 2:
                    print(f"Precisión Top-3: {resultados[2]:.4f}")
            else:
                print("Primero entrena el modelo o carga datos de validación.")
                
        elif opcion == '4':
            if entrenador.historial:
                entrenador.graficar_historial_entrenamiento()
            else:
                print("No hay historial de entrenamiento disponible.")
                
        elif opcion == '5':
            if predictor is None or predictor.modelo is None:
                print("Primero carga un modelo (opción 2).")
                continue
                
            ruta_imagen = input("Ruta de la imagen: ").strip()
            if os.path.exists(ruta_imagen):
                resultados = predictor.predecir_imagen(ruta_imagen)
                if resultados:
                    print("\n--- Predicciones ---")
                    for i, (caracter, prob) in enumerate(resultados, 1):
                        print(f"{i}. {caracter}: {prob:.4f}")                        
                    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        plt.figure(figsize=(6, 6))
                        plt.imshow(img, cmap='gray')
                        plt.title(f'Predicción: {resultados[0][0]} ({resultados[0][1]:.3f})')
                        plt.axis('off')
                        plt.show()
            else:
                print("Archivo no encontrado.")
                
        elif opcion == '6':
            print("\n--- Información del Dataset ---")
            print(f"Tamaño de imagen: {configuracion.TAMAÑO_IMG}x{configuracion.TAMAÑO_IMG}")
            print(f"Tamaño de parche: {configuracion.TAMAÑO_PARCHE}x{configuracion.TAMAÑO_PARCHE}")
            print(f"Número de parches: {configuracion.NUM_PARCHES}")
            print(f"Clases objetivo: {configuracion.CLASES_OBJETIVO}")
            print("\nEste sistema puede usar:")
            print("- Imágenes locales en ./datos_chinos/")
            print("- Dataset sintético generado automáticamente")
            print("- CASIA-HWDB u otros datasets estándar (con modificaciones)")
            print("\nCORRECCIÓN APLICADA:")
            print("- Capas personalizadas ahora se registran correctamente")
            print("- get_config() y from_config() implementados")
            print("- @keras.utils.register_keras_serializable() decorators añadidos")
            
        elif opcion == '7':
            print("¡Hasta luego!")
            break
            
        else:
            print("Opción no válida. Por favor selecciona 1-7.")

if __name__ == "__main__":
    principal()