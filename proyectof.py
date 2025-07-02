import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from numba import cuda, float32, int32
import math
from scipy import ndimage
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import os
import pickle
from tqdm import tqdm

@cuda.jit
def kernel_sobel_x(imagen, salida, alto, ancho):
    i, j = cuda.grid(2)
    if i >= 1 and i < alto - 1 and j >= 1 and j < ancho - 1:
        resultado = (-1 * imagen[i-1, j-1] + 0 * imagen[i-1, j] + 1 * imagen[i-1, j+1] +
                    -2 * imagen[i, j-1]   + 0 * imagen[i, j]   + 2 * imagen[i, j+1] +
                    -1 * imagen[i+1, j-1] + 0 * imagen[i+1, j] + 1 * imagen[i+1, j+1])
        salida[i, j] = abs(resultado)

@cuda.jit
def kernel_sobel_y(imagen, salida, alto, ancho):
    i, j = cuda.grid(2)
    if i >= 1 and i < alto - 1 and j >= 1 and j < ancho - 1:
        resultado = (-1 * imagen[i-1, j-1] + -2 * imagen[i-1, j] + -1 * imagen[i-1, j+1] +
                     0 * imagen[i, j-1]   +  0 * imagen[i, j]   +  0 * imagen[i, j+1] +
                     1 * imagen[i+1, j-1] +  2 * imagen[i+1, j] +  1 * imagen[i+1, j+1])
        salida[i, j] = abs(resultado)

@cuda.jit
def kernel_laplaciano(imagen, salida, alto, ancho):
    i, j = cuda.grid(2)
    if i >= 1 and i < alto - 1 and j >= 1 and j < ancho - 1:
        resultado = (0 * imagen[i-1, j-1] + -1 * imagen[i-1, j] + 0 * imagen[i-1, j+1] +
                    -1 * imagen[i, j-1]   +  4 * imagen[i, j]   + -1 * imagen[i, j+1] +
                     0 * imagen[i+1, j-1] + -1 * imagen[i+1, j] + 0 * imagen[i+1, j+1])
        salida[i, j] = abs(resultado)

@cuda.jit
def kernel_desenfoque_gaussiano(imagen, salida, alto, ancho):
    i, j = cuda.grid(2)
    if i >= 1 and i < alto - 1 and j >= 1 and j < ancho - 1:
        resultado = (1 * imagen[i-1, j-1] + 2 * imagen[i-1, j] + 1 * imagen[i-1, j+1] +
                    2 * imagen[i, j-1]   + 4 * imagen[i, j]   + 2 * imagen[i, j+1] +
                    1 * imagen[i+1, j-1] + 2 * imagen[i+1, j] + 1 * imagen[i+1, j+1]) / 16.0
        salida[i, j] = resultado

@cuda.jit
def kernel_enfoque(imagen, salida, alto, ancho):
    i, j = cuda.grid(2)
    if i >= 1 and i < alto - 1 and j >= 1 and j < ancho - 1:
        resultado = (0 * imagen[i-1, j-1] + -1 * imagen[i-1, j] + 0 * imagen[i-1, j+1] +
                    -1 * imagen[i, j-1]   +  5 * imagen[i, j]   + -1 * imagen[i, j+1] +
                     0 * imagen[i+1, j-1] + -1 * imagen[i+1, j] + 0 * imagen[i+1, j+1])
        salida[i, j] = max(0, min(255, resultado))

class GeneradorDatos:
    def __init__(self):
        self.tipos_ruido = ['gaussiano', 'sal_pimienta', 'desenfoque', 'compresion']
        
    def generar_imagen_limpia(self, tamano=(128, 128)):
        imagen = np.zeros(tamano, dtype=np.float32)
        cv2.rectangle(imagen, (20, 20), (60, 60), 0.8, -1)
        cv2.circle(imagen, (90, 40), 25, 0.6, -1)
        cv2.ellipse(imagen, (40, 90), (30, 15), 45, 0, 360, 0.9, -1)
        
        for i in range(0, tamano[0], 8):
            cv2.line(imagen, (i, 70), (i, 110), 0.4, 1)
        
        return imagen
    
    def anadir_ruido(self, imagen, tipo_ruido, intensidad=0.1):
        imagen_ruidosa = imagen.copy()
        
        if tipo_ruido == 'gaussiano':
            ruido = np.random.normal(0, intensidad, imagen.shape)
            imagen_ruidosa = np.clip(imagen + ruido, 0, 1)   
        elif tipo_ruido == 'sal_pimienta':
            mascara = np.random.random(imagen.shape) < intensidad
            imagen_ruidosa[mascara] = np.random.choice([0, 1], size=np.sum(mascara))
        elif tipo_ruido == 'desenfoque':
            kernel_size = max(3, int(intensidad * 15))
            if kernel_size % 2 == 0:
                kernel_size += 1
            imagen_ruidosa = cv2.GaussianBlur(imagen, (kernel_size, kernel_size), 0)
        elif tipo_ruido == 'compresion':
            imagen_uint8 = (imagen * 255).astype(np.uint8)
            quality = max(10, int(100 - intensidad * 90))
            _, encoded = cv2.imencode('.jpg', imagen_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
            imagen_ruidosa = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return np.clip(imagen_ruidosa, 0, 1)
    
    def generar_lote_entrenamiento(self, tamano_lote, tamano_imagen=(128, 128)):
        imagenes_limpias = []
        imagenes_ruidosas = []
        for _ in range(tamano_lote):
            img_limpia = self.generar_imagen_limpia(tamano_imagen)            
            tipo_ruido = np.random.choice(self.tipos_ruido)
            intensidad = np.random.uniform(0.05, 0.3)
            img_ruidosa = self.anadir_ruido(img_limpia, tipo_ruido, intensidad)
            imagenes_limpias.append(img_limpia)
            imagenes_ruidosas.append(img_ruidosa)
        return np.array(imagenes_ruidosas), np.array(imagenes_limpias)

class TransformadorMejoraImagen:    
    def __init__(self, tamano_parche=16, dimension_modelo=256, num_cabezas=8, num_capas=4, learning_rate=0.001):
        self.tamano_parche = tamano_parche
        self.dimension_modelo = dimension_modelo
        self.num_cabezas = num_cabezas
        self.num_capas = num_capas
        self.learning_rate = learning_rate        
        np.random.seed(42)
        self._inicializar_pesos()        
        self._inicializar_optimizador()        
        self.historial_perdida = []
        self.historial_metricas = []
        
    def _inicializar_pesos(self):
        self.incrustacion_parche = np.random.normal(0, 0.02, 
            (self.tamano_parche * self.tamano_parche, self.dimension_modelo))
        self.incrustacion_posicion = np.random.normal(0, 0.02, (1000, self.dimension_modelo))
        self.pesos_atencion = []
        for _ in range(self.num_capas):
            pesos_capa = {
                'consulta': np.random.normal(0, 0.02, (self.dimension_modelo, self.dimension_modelo)),
                'clave': np.random.normal(0, 0.02, (self.dimension_modelo, self.dimension_modelo)),
                'valor': np.random.normal(0, 0.02, (self.dimension_modelo, self.dimension_modelo)),
                'salida': np.random.normal(0, 0.02, (self.dimension_modelo, self.dimension_modelo)),
                'ffn_w1': np.random.normal(0, 0.02, (self.dimension_modelo, self.dimension_modelo * 4)),
                'ffn_w2': np.random.normal(0, 0.02, (self.dimension_modelo * 4, self.dimension_modelo)),
                'gamma1': np.ones(self.dimension_modelo),
                'beta1': np.zeros(self.dimension_modelo),
                'gamma2': np.ones(self.dimension_modelo),
                'beta2': np.zeros(self.dimension_modelo)
            }
            self.pesos_atencion.append(pesos_capa)        
        self.cabeza_reconstruccion = np.random.normal(0, 0.02, 
            (self.dimension_modelo, self.tamano_parche * self.tamano_parche))
    
    def _inicializar_optimizador(self):
        self.momentos_m = {}
        self.momentos_v = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.paso = 0
        self.momentos_m['patch_embed'] = np.zeros_like(self.incrustacion_parche)
        self.momentos_v['patch_embed'] = np.zeros_like(self.incrustacion_parche)
        self.momentos_m['pos_embed'] = np.zeros_like(self.incrustacion_posicion)
        self.momentos_v['pos_embed'] = np.zeros_like(self.incrustacion_posicion)

        for i in range(self.num_capas):
            for key in self.pesos_atencion[i]:
                self.momentos_m[f'layer_{i}_{key}'] = np.zeros_like(self.pesos_atencion[i][key])
                self.momentos_v[f'layer_{i}_{key}'] = np.zeros_like(self.pesos_atencion[i][key])
        
        self.momentos_m['recon_head'] = np.zeros_like(self.cabeza_reconstruccion)
        self.momentos_v['recon_head'] = np.zeros_like(self.cabeza_reconstruccion)
    
    def extraer_parches(self, imagen):
        alto, ancho = imagen.shape
        parches = []
        posiciones = []
        alto_util = (alto // self.tamano_parche) * self.tamano_parche
        ancho_util = (ancho // self.tamano_parche) * self.tamano_parche
        pos_idx = 0
        for i in range(0, alto_util, self.tamano_parche):
            for j in range(0, ancho_util, self.tamano_parche):
                parche = imagen[i:i+self.tamano_parche, j:j+self.tamano_parche]
                parches.append(parche.flatten())
                posiciones.append(pos_idx)
                pos_idx += 1
        return np.array(parches), posiciones
    
    def mecanismo_atencion(self, x, pesos, calcular_gradientes=False):
        seq_len, d_model = x.shape
        d_k = d_model // self.num_cabezas
        Q = np.dot(x, pesos['consulta'])
        K = np.dot(x, pesos['clave'])
        V = np.dot(x, pesos['valor'])
        Q = Q.reshape(seq_len, self.num_cabezas, d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_cabezas, d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_cabezas, d_k).transpose(1, 0, 2)
        outputs = []
        attention_weights_all = []
        for h in range(self.num_cabezas):
            scores = np.dot(Q[h], K[h].T) / math.sqrt(d_k)
            attention_weights = self.softmax(scores)
            attended = np.dot(attention_weights, V[h])
            outputs.append(attended)
            attention_weights_all.append(attention_weights)    
        multi_head_output = np.concatenate(outputs, axis=-1)  
        salida = np.dot(multi_head_output, pesos['salida'])
        if calcular_gradientes:
            return salida, {
                'Q': Q, 'K': K, 'V': V,
                'attention_weights': attention_weights_all,
                'multi_head_output': multi_head_output,
                'input': x
            }
        return salida
    
    def feed_forward(self, x, pesos):
        h1 = np.dot(x, pesos['ffn_w1'])
        h1 = np.maximum(0, h1)  # ReLU
        h2 = np.dot(h1, pesos['ffn_w2'])
        return h2
    
    def normalizacion_capa(self, x, gamma, beta):
        media = np.mean(x, axis=-1, keepdims=True)
        varianza = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - media) / np.sqrt(varianza + self.epsilon)
        return gamma * x_norm + beta
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, imagen, calcular_gradientes=False):
        if imagen.shape[0] < self.tamano_parche or imagen.shape[1] < self.tamano_parche:
            raise ValueError(f"Imagen muy pequeña. Mínimo: {self.tamano_parche}x{self.tamano_parche}")
        parches, posiciones = self.extraer_parches(imagen)
        if len(parches) == 0:
            raise ValueError("No se pudieron extraer parches de la imagen")
        x = np.dot(parches, self.incrustacion_parche)
        for i, pos in enumerate(posiciones):
            if i < len(x) and pos < len(self.incrustacion_posicion):
                x[i] += self.incrustacion_posicion[pos]
        activaciones = [x.copy()] if calcular_gradientes else None
    
        for i, pesos_capa in enumerate(self.pesos_atencion):
            try:
                if calcular_gradientes:
                    atencion_out, atencion_cache = self.mecanismo_atencion(x, pesos_capa, True)
                else:
                    atencion_out = self.mecanismo_atencion(x, pesos_capa, False)
                x = self.normalizacion_capa(x + atencion_out, pesos_capa['gamma1'], pesos_capa['beta1'])
                ffn_out = self.feed_forward(x, pesos_capa)
                x = self.normalizacion_capa(x + ffn_out, pesos_capa['gamma2'], pesos_capa['beta2'])
                if calcular_gradientes:
                    activaciones.append(x.copy())
            except Exception as e:
                print(f"Error en capa {i}: {e}")
                if calcular_gradientes:
                    activaciones.append(x.copy())
        try:
            parches_reconstruidos = np.dot(x, self.cabeza_reconstruccion)
            imagen_reconstruida = self.reconstruir_imagen(parches_reconstruidos, imagen.shape)
        except Exception as e:
            print(f"Error en reconstrucción: {e}")
            imagen_reconstruida = imagen.copy()
            parches_reconstruidos = parches
        if calcular_gradientes:
            return imagen_reconstruida, activaciones, parches_reconstruidos
        return imagen_reconstruida
    
    def reconstruir_imagen(self, parches_reconstruidos, forma_original):
        alto, ancho = forma_original
        alto_util = (alto // self.tamano_parche) * self.tamano_parche
        ancho_util = (ancho // self.tamano_parche) * self.tamano_parche
        imagen_reconstruida = np.zeros((alto_util, ancho_util), dtype=np.float32)
        indice_parche = 0
        for i in range(0, alto_util, self.tamano_parche):
            for j in range(0, ancho_util, self.tamano_parche):
                if indice_parche < len(parches_reconstruidos):
                    parche = parches_reconstruidos[indice_parche].reshape(
                        self.tamano_parche, self.tamano_parche)
                    imagen_reconstruida[i:i+self.tamano_parche, j:j+self.tamano_parche] = parche
                    indice_parche += 1
        if (alto_util, ancho_util) != (alto, ancho):
            imagen_reconstruida = cv2.resize(imagen_reconstruida, (ancho, alto))
        return np.clip(imagen_reconstruida, 0, 1)
    
    def calcular_perdida(self, prediccion, objetivo):
        if prediccion.shape != objetivo.shape:
            prediccion = cv2.resize(prediccion, objetivo.shape[::-1])
        mse_loss = np.mean((prediccion - objetivo) ** 2)
        try:
            if prediccion.shape[1] > 1:  
                grad_pred_x = np.diff(prediccion, axis=1)
                grad_obj_x = np.diff(objetivo, axis=1)
                grad_loss_x = np.mean((grad_pred_x - grad_obj_x) ** 2)
            else:
                grad_loss_x = 0
            if prediccion.shape[0] > 1:  
                grad_pred_y = np.diff(prediccion, axis=0)
                grad_obj_y = np.diff(objetivo, axis=0)
                grad_loss_y = np.mean((grad_pred_y - grad_obj_y) ** 2)
            else:
                grad_loss_y = 0
            grad_loss = grad_loss_x + grad_loss_y
        except:
            grad_loss = 0  
        smooth_loss = 0
        try:
            if prediccion.shape[0] > 2 and prediccion.shape[1] > 2:
                laplacian = cv2.Laplacian(prediccion, cv2.CV_32F)
                smooth_loss = np.mean(laplacian ** 2)
        except:
            smooth_loss = 0
        total_loss = mse_loss + 0.1 * grad_loss + 0.01 * smooth_loss
        return total_loss, mse_loss, grad_loss
    
    def backward(self, prediccion, objetivo, activaciones, parches_reconstruidos):
        self.paso += 1
        batch_size = prediccion.size
        grad_prediccion = 2 * (prediccion - objetivo) / batch_size
        parches_grad, _ = self.extraer_parches(grad_prediccion)
        if len(parches_grad) != len(parches_reconstruidos):
            min_len = min(len(parches_grad), len(parches_reconstruidos))
            parches_grad = parches_grad[:min_len]
            parches_reconstruidos = parches_reconstruidos[:min_len]
        if len(activaciones) > 0:
            ultima_activacion = activaciones[-1]
            if len(ultima_activacion) != len(parches_grad):
                min_len = min(len(ultima_activacion), len(parches_grad))
                ultima_activacion = ultima_activacion[:min_len]
                parches_grad = parches_grad[:min_len]
            grad_cabeza = np.dot(ultima_activacion.T, parches_grad)
            grad_cabeza = np.clip(grad_cabeza, -1.0, 1.0)
            self._actualizar_adam('recon_head', self.cabeza_reconstruccion, grad_cabeza)

    
    def _actualizar_adam(self, nombre, pesos, gradientes):
        self.momentos_m[nombre] = (self.beta1 * self.momentos_m[nombre] + 
                                  (1 - self.beta1) * gradientes)
        self.momentos_v[nombre] = (self.beta2 * self.momentos_v[nombre] + 
                                  (1 - self.beta2) * gradientes ** 2)
        m_hat = self.momentos_m[nombre] / (1 - self.beta1 ** self.paso)
        v_hat = self.momentos_v[nombre] / (1 - self.beta2 ** self.paso)
        pesos -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def entrenar_lote(self, imagenes_entrada, imagenes_objetivo):
        perdida_total = 0
        mse_total = 0
        grad_total = 0
        for i in range(len(imagenes_entrada)):
            prediccion, activaciones, parches_reconstruidos = self.forward(
                imagenes_entrada[i], calcular_gradientes=True)
            perdida, mse, grad_loss = self.calcular_perdida(prediccion, imagenes_objetivo[i])
            self.backward(prediccion, imagenes_objetivo[i], activaciones, parches_reconstruidos)
            perdida_total += perdida
            mse_total += mse
            grad_total += grad_loss
        return (perdida_total / len(imagenes_entrada), 
                mse_total / len(imagenes_entrada),
                grad_total / len(imagenes_entrada))

    def entrenar(self, generador_datos, num_epocas=100, tamano_lote=4, tamano_imagen=(128, 128)):
        print(f" Iniciando entrenamiento por {num_epocas} épocas")
        print(f"Tamaño de lote: {tamano_lote}, Tamaño de imagen: {tamano_imagen}")
        print("=" * 60)
        
        for epoca in range(num_epocas):
            epoca_perdida = 0
            epoca_mse = 0
            epoca_grad = 0
            num_lotes = 10 
            
            with tqdm(range(num_lotes), desc=f'Época {epoca+1}/{num_epocas}') as pbar:
                for lote in pbar:
                    imgs_entrada, imgs_objetivo = generador_datos.generar_lote_entrenamiento(
                        tamano_lote, tamano_imagen)
                    perdida_lote, mse_lote, grad_lote = self.entrenar_lote(imgs_entrada, imgs_objetivo)
                    epoca_perdida += perdida_lote
                    epoca_mse += mse_lote
                    epoca_grad += grad_lote
                    pbar.set_postfix({
                        'Loss': f'{perdida_lote:.4f}',
                        'MSE': f'{mse_lote:.4f}',
                        'Grad': f'{grad_lote:.4f}'
                    })
            epoca_perdida /= num_lotes
            epoca_mse /= num_lotes
            epoca_grad /= num_lotes
            
            self.historial_perdida.append(epoca_perdida)
            self.historial_metricas.append({
                'mse': epoca_mse,
                'grad_loss': epoca_grad,
                'epoca': epoca + 1
            })
    
            if (epoca + 1) % 10 == 0:
                print(f"\nÉpoca {epoca+1}: Loss={epoca_perdida:.4f}, MSE={epoca_mse:.4f}, Grad={epoca_grad:.4f}")
                self._evaluar_muestra(generador_datos, tamano_imagen)
        
        print("\n Entrenamiento completado!")
    
    def _evaluar_muestra(self, generador_datos, tamano_imagen):
        img_entrada, img_objetivo = generador_datos.generar_lote_entrenamiento(1, tamano_imagen)
        img_mejorada = self.forward(img_entrada[0])
        psnr = peak_signal_noise_ratio(img_objetivo[0], img_mejorada, data_range=1.0)
        ssim = structural_similarity(img_objetivo[0], img_mejorada, data_range=1.0)
        
        print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    def guardar_modelo(self, ruta):
        modelo_data = {
            'incrustacion_parche': self.incrustacion_parche,
            'incrustacion_posicion': self.incrustacion_posicion,
            'pesos_atencion': self.pesos_atencion,
            'cabeza_reconstruccion': self.cabeza_reconstruccion,
            'historial_perdida': self.historial_perdida,
            'historial_metricas': self.historial_metricas,
            'config': {
                'tamano_parche': self.tamano_parche,
                'dimension_modelo': self.dimension_modelo,
                'num_cabezas': self.num_cabezas,
                'num_capas': self.num_capas,
                'learning_rate': self.learning_rate
            }
        }
        
        with open(ruta, 'wb') as f:
            pickle.dump(modelo_data, f)
        print(f" Modelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta):
        with open(ruta, 'rb') as f:
            modelo_data = pickle.load(f)
        self.incrustacion_parche = modelo_data['incrustacion_parche']
        self.incrustacion_posicion = modelo_data['incrustacion_posicion']
        self.pesos_atencion = modelo_data['pesos_atencion']
        self.cabeza_reconstruccion = modelo_data['cabeza_reconstruccion']
        self.historial_perdida = modelo_data['historial_perdida']
        self.historial_metricas = modelo_data['historial_metricas']
        print(f" Modelo cargado desde: {ruta}")
    
    def graficar_entrenamiento(self):
        if not self.historial_perdida:
            print("No hay historial de entrenamiento para graficar")
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(self.historial_perdida)
        axes[0, 0].set_title('Pérdida Total')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Pérdida')
        axes[0, 0].grid(True)
        mse_values = [m['mse'] for m in self.historial_metricas]
        axes[0, 1].plot(mse_values)
        axes[0, 1].set_title('Error Cuadrático Medio')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].grid(True)
        grad_values = [m['grad_loss'] for m in self.historial_metricas]
        axes[1, 0].plot(grad_values)
        axes[1, 0].set_title('Pérdida de Gradiente')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Grad Loss')
        axes[1, 0].grid(True)
        axes[1, 1].plot([self.learning_rate] * len(self.historial_perdida))
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig('entrenamiento_progreso.png', dpi=300, bbox_inches='tight')
        plt.show()

class ProcesadorImagenCUDA:
    def __init__(self):
        self.transformador = TransformadorMejoraImagen()
        self.generador_datos = GeneradorDatos()
        self.filtros = {
            'sobel_x': kernel_sobel_x,
            'sobel_y': kernel_sobel_y,
            'laplaciano': kernel_laplaciano,
            'gaussiano': kernel_desenfoque_gaussiano,
            'enfoque': kernel_enfoque
        }
    
    def entrenar_transformador(self, num_epocas=50, tamano_lote=4):
        print(" Entrenando Transformer")
        print("=" * 60)
        self.transformador.entrenar(
            self.generador_datos, 
            num_epocas=num_epocas, 
            tamano_lote=tamano_lote
        )
        self.transformador.graficar_entrenamiento()
        self.transformador.guardar_modelo('transformer_modelo.pkl')
    
    def demostrar_mejora_imagen(self, ruta_imagen=None):
        print("\n Demostrando Mejora de Imagen")
        print("=" * 40)
        if ruta_imagen and os.path.exists(ruta_imagen):
            imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            imagen_original = cv2.resize(imagen_original, (128, 128))
            imagen_original = imagen_original.astype(np.float32) / 255.0
            print(f"Usando imagen: {ruta_imagen}")
        else:
            imagen_original = self.generador_datos.generar_imagen_limpia((128, 128))
            print("Usando imagen sintética")
        imagen_ruido_gaussiano = self.generador_datos.anadir_ruido(imagen_original, 'gaussiano', 0.2)
        imagen_desenfoque = self.generador_datos.anadir_ruido(imagen_original, 'desenfoque', 0.3)
        imagen_compresion = self.generador_datos.anadir_ruido(imagen_original, 'compresion', 0.4)
        print("Procesando con Transformer...")
        tiempo_inicio = time.time()
        imagen_mejorada_ruido = self.transformador.forward(imagen_ruido_gaussiano)
        imagen_mejorada_desenfoque = self.transformador.forward(imagen_desenfoque)
        imagen_mejorada_compresion = self.transformador.forward(imagen_compresion)
        tiempo_transformer = time.time() - tiempo_inicio
        metricas = self._calcular_metricas_mejora(
            imagen_original,
            [imagen_ruido_gaussiano, imagen_desenfoque, imagen_compresion],
            [imagen_mejorada_ruido, imagen_mejorada_desenfoque, imagen_mejorada_compresion],
            ['Ruido Gaussiano', 'Desenfoque', 'Compresión']
        )
        print(f"Tiempo de procesamiento: {tiempo_transformer:.4f}s")
        print("\nMétricas de Mejora:")
        for nombre, metrica in metricas.items():
            print(f"{nombre}:")
            print(f"  PSNR: {metrica['psnr_antes']:.2f} → {metrica['psnr_despues']:.2f} dB (+{metrica['mejora_psnr']:.2f})")
            print(f"  SSIM: {metrica['ssim_antes']:.4f} → {metrica['ssim_despues']:.4f} (+{metrica['mejora_ssim']:.4f})")
        
        self._visualizar_mejoras(
            imagen_original,
            [imagen_ruido_gaussiano, imagen_desenfoque, imagen_compresion],
            [imagen_mejorada_ruido, imagen_mejorada_desenfoque, imagen_mejorada_compresion],
            ['Ruido Gaussiano', 'Desenfoque', 'Compresión']
        )
        return metricas
    
    def _calcular_metricas_mejora(self, original, degradadas, mejoradas, nombres):
        metricas = {}
        for i, nombre in enumerate(nombres):
            psnr_antes = peak_signal_noise_ratio(original, degradadas[i], data_range=1.0)
            ssim_antes = structural_similarity(original, degradadas[i], data_range=1.0)
            psnr_despues = peak_signal_noise_ratio(original, mejoradas[i], data_range=1.0)
            ssim_despues = structural_similarity(original, mejoradas[i], data_range=1.0)
            metricas[nombre] = {
                'psnr_antes': psnr_antes,
                'psnr_despues': psnr_despues,
                'mejora_psnr': psnr_despues - psnr_antes,
                'ssim_antes': ssim_antes,
                'ssim_despues': ssim_despues,
                'mejora_ssim': ssim_despues - ssim_antes
            }
        return metricas
    
    def _visualizar_mejoras(self, original, degradadas, mejoradas, nombres):
        fig, axes = plt.subplots(4, len(nombres) + 1, figsize=(16, 12))
        axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        for i in range(1, 4):
            axes[i, 0].axis('off')
        for j, nombre in enumerate(nombres):
            col = j + 1
            axes[0, col].imshow(degradadas[j], cmap='gray', vmin=0, vmax=1)
            axes[0, col].set_title(f'{nombre}\n(Degradada)', fontsize=10)
            axes[0, col].axis('off')
            axes[1, col].imshow(mejoradas[j], cmap='gray', vmin=0, vmax=1)
            axes[1, col].set_title(f'{nombre}\n(Mejorada)', fontsize=10)
            axes[1, col].axis('off')
            diff_degradada = np.abs(original - degradadas[j])
            axes[2, col].imshow(diff_degradada, cmap='hot', vmin=0, vmax=0.5)
            axes[2, col].set_title('Diff: Orig-Degradada', fontsize=9)
            axes[2, col].axis('off')
            diff_mejorada = np.abs(original - mejoradas[j])
            axes[3, col].imshow(diff_mejorada, cmap='hot', vmin=0, vmax=0.5)
            axes[3, col].set_title('Diff: Orig-Mejorada', fontsize=9)
            axes[3, col].axis('off')
        plt.suptitle('Resultados de Mejora con Transformer', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('mejora_imagenes_transformer.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def aplicar_filtro_gpu(self, imagen, nombre_filtro):
        if nombre_filtro not in self.filtros:
            raise ValueError(f"Filtro {nombre_filtro} no disponible")
        alto, ancho = imagen.shape
        hilos_por_bloque = (16, 16)
        bloques_por_cuadricula_x = math.ceil(alto / hilos_por_bloque[0])
        bloques_por_cuadricula_y = math.ceil(ancho / hilos_por_bloque[1])
        bloques_por_cuadricula = (bloques_por_cuadricula_x, bloques_por_cuadricula_y)
        d_imagen = cuda.to_device(imagen.astype(np.float32))
        d_salida = cuda.device_array((alto, ancho), dtype=np.float32)
        tiempo_inicio = time.time()
        self.filtros[nombre_filtro][bloques_por_cuadricula, hilos_por_bloque](
            d_imagen, d_salida, alto, ancho
        )
        cuda.synchronize()
        tiempo_gpu = time.time() - tiempo_inicio
        resultado = d_salida.copy_to_host()
        return resultado, tiempo_gpu
    
    def aplicar_filtro_cpu(self, imagen, nombre_filtro):
        tiempo_inicio = time.time()
        if nombre_filtro == 'sobel_x':
            nucleo = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            resultado = np.abs(ndimage.convolve(imagen.astype(np.float32), nucleo))
        elif nombre_filtro == 'sobel_y':
            nucleo = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            resultado = np.abs(ndimage.convolve(imagen.astype(np.float32), nucleo))
        elif nombre_filtro == 'laplaciano':
            nucleo = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            resultado = np.abs(ndimage.convolve(imagen.astype(np.float32), nucleo))
        elif nombre_filtro == 'gaussiano':
            nucleo = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
            resultado = ndimage.convolve(imagen.astype(np.float32), nucleo)
        elif nombre_filtro == 'enfoque':
            nucleo = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            resultado = ndimage.convolve(imagen.astype(np.float32), nucleo)
            resultado = np.clip(resultado, 0, 255)
        else:
            raise ValueError(f"Filtro {nombre_filtro} no disponible")
        tiempo_cpu = time.time() - tiempo_inicio
        return resultado, tiempo_cpu
    
    def comparar_todos(self, ruta_imagen=None):
        print(" Comparación Completa: Filtros CUDA vs Transformer")
        print("=" * 60)
        if ruta_imagen and os.path.exists(ruta_imagen):
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            imagen = cv2.resize(imagen, (128, 128))
            print(f"Usando imagen: {ruta_imagen}")
        else:
            imagen_limpia = self.generador_datos.generar_imagen_limpia((128, 128))
            imagen = (imagen_limpia * 255).astype(np.uint8)
            print("Usando imagen sintética")
        print(f"Tamaño de imagen: {imagen.shape}")
        resultados_filtros = {}
        for nombre_filtro in self.filtros.keys():
            print(f"\n--- Filtro CUDA: {nombre_filtro.upper()} ---")
            try:
                resultado_gpu, tiempo_gpu = self.aplicar_filtro_gpu(imagen, nombre_filtro)
                resultado_cpu, tiempo_cpu = self.aplicar_filtro_cpu(imagen, nombre_filtro)
                aceleracion = tiempo_cpu / tiempo_gpu if tiempo_gpu > 0 else float('inf')
                error = mean_squared_error(resultado_cpu.flatten(), resultado_gpu.flatten())
                resultados_filtros[nombre_filtro] = {
                    'tiempo_gpu': tiempo_gpu,
                    'tiempo_cpu': tiempo_cpu,
                    'aceleracion': aceleracion,
                    'error': error,
                    'resultado_gpu': resultado_gpu,
                    'resultado_cpu': resultado_cpu
                }
                print(f"Tiempo GPU: {tiempo_gpu:.6f}s | CPU: {tiempo_cpu:.6f}s | Aceleración: {aceleracion:.2f}x")
            except Exception as e:
                print(f"Error en filtro {nombre_filtro}: {e}")
        print(f"\n--- TRANSFORMER DE MEJORA ---")
        try:
            imagen_normalizada = imagen.astype(np.float32) / 255.0
            imagen_degradada = self.generador_datos.anadir_ruido(imagen_normalizada, 'gaussiano', 0.15)
            tiempo_inicio = time.time()
            imagen_mejorada = self.transformador.forward(imagen_degradada)
            tiempo_transformer = time.time() - tiempo_inicio
            psnr_antes = peak_signal_noise_ratio(imagen_normalizada, imagen_degradada, data_range=1.0)
            psnr_despues = peak_signal_noise_ratio(imagen_normalizada, imagen_mejorada, data_range=1.0)
            ssim_antes = structural_similarity(imagen_normalizada, imagen_degradada, data_range=1.0)
            ssim_despues = structural_similarity(imagen_normalizada, imagen_mejorada, data_range=1.0)
            print(f"Tiempo Transformer: {tiempo_transformer:.6f}s")
            print(f"PSNR: {psnr_antes:.2f} → {psnr_despues:.2f} dB (mejora: {psnr_despues-psnr_antes:.2f})")
            print(f"SSIM: {ssim_antes:.4f} → {ssim_despues:.4f} (mejora: {ssim_despues-ssim_antes:.4f})")
            resultados_filtros['transformer'] = {
                'tiempo': tiempo_transformer,
                'psnr_mejora': psnr_despues - psnr_antes,
                'ssim_mejora': ssim_despues - ssim_antes,
                'imagen_degradada': imagen_degradada,
                'imagen_mejorada': imagen_mejorada
            }
            
        except Exception as e:
            print(f"Error en Transformer: {e}")
        self._visualizar_comparacion_completa(imagen, resultados_filtros)
        return resultados_filtros
    
    def _visualizar_comparacion_completa(self, imagen_original, resultados):
        filtros_cuda = [k for k in resultados.keys() if k != 'transformer']
        num_filas = 3
        num_cols = len(filtros_cuda) + 2 
        fig, axes = plt.subplots(num_filas, num_cols, figsize=(20, 12))
        axes[0, 0].imshow(imagen_original, cmap='gray')
        axes[0, 0].set_title('Original', fontweight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')
        col = 1
        for nombre_filtro in filtros_cuda:
            if nombre_filtro in resultados:
                axes[0, col].imshow(resultados[nombre_filtro]['resultado_gpu'], cmap='gray')
                axes[0, col].set_title(f'{nombre_filtro.upper()}\n(GPU: {resultados[nombre_filtro]["tiempo_gpu"]:.4f}s)', fontsize=9)
                axes[0, col].axis('off')
                axes[1, col].imshow(resultados[nombre_filtro]['resultado_cpu'], cmap='gray')
                axes[1, col].set_title(f'{nombre_filtro.upper()}\n(CPU: {resultados[nombre_filtro]["tiempo_cpu"]:.4f}s)', fontsize=9)
                axes[1, col].axis('off')
                axes[2, col].text(0.5, 0.5, f'Aceleración:\n{resultados[nombre_filtro]["aceleracion"]:.1f}x\n\nError MSE:\n{resultados[nombre_filtro]["error"]:.4f}', 
                                ha='center', va='center', transform=axes[2, col].transAxes, fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[2, col].axis('off')
            col += 1
        
        if 'transformer' in resultados:
            axes[0, -1].imshow(resultados['transformer']['imagen_degradada'], cmap='gray')
            axes[0, -1].set_title(f'Transformer\n(Degradada)', fontsize=9)
            axes[0, -1].axis('off')
            axes[1, -1].imshow(resultados['transformer']['imagen_mejorada'], cmap='gray')
            axes[1, -1].set_title(f'Transformer\n(Mejorada: {resultados["transformer"]["tiempo"]:.4f}s)', fontsize=9)
            axes[1, -1].axis('off')
            axes[2, -1].text(0.5, 0.5, f'PSNR mejora:\n{resultados["transformer"]["psnr_mejora"]:.2f} dB\n\nSSIM mejora:\n{resultados["transformer"]["ssim_mejora"]:.4f}', 
                            ha='center', va='center', transform=axes[2, -1].transAxes, fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[2, -1].axis('off')
        plt.suptitle('Comparación: Filtros CUDA vs Transformer de Mejora', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('comparacion_completa.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generar_imagen_sintetica(self, tamano=(512, 512)):
        return self.generador_datos.generar_imagen_limpia(tamano)

def principal():
    print(" Proyecto de Computo paralelo")
    print("CUDA + Transformer Entrenable para Mejora de Imágenes")
    print("=" * 70)
    try:
        cuda.detect()
        print(" CUDA disponible")
        print(f"Dispositivos CUDA detectados: {len(cuda.gpus)}")
        for i, gpu in enumerate(cuda.gpus):
            print(f"  GPU {i}: {gpu.name}")
    except Exception as e:
        print(f" Error CUDA: {e}")
        return
    procesador = ProcesadorImagenCUDA()
    print("\n Opciones disponibles:")
    print("1. Entrenar Transformer para mejora de imágenes")
    print("2. Demostrar mejora de imagen")
    print("3. Comparación completa: CUDA vs Transformer")
    print("4. Cargar modelo pre-entrenado y demostrar")
    try:
        opcion = input("\nSelecciona una opción: ").strip()
        if opcion == "1":
            print("\n🎓 Iniciando entrenamiento...")
            num_epocas = int(input("Número de épocas: ") or "30")
            tamano_lote = int(input("Tamaño de lote: ") or "2")
            procesador.entrenar_transformador(num_epocas, tamano_lote)
        elif opcion == "2":
            print("\n📸 Demostrando mejora de imagen...")
            ruta_imagen = input("Ruta de imagen (Enter para imagen sintética): ").strip() or None
            try:
                procesador.transformador.cargar_modelo('transformer_mejorado.pkl')
                procesador.demostrar_mejora_imagen(ruta_imagen)
            except FileNotFoundError:
                print(" No se encontró modelo entrenado. Ejecuta primero la opción 1.")
        elif opcion == "3":
            print("\n🔬 Comparación completa...")
            ruta_imagen = input("Ruta de imagen (Enter para imagen sintética): ").strip() or None
            try:
                procesador.transformador.cargar_modelo('transformer_mejorado.pkl')
                print(" Modelo cargado para comparación")
            except FileNotFoundError:
                print("  No se encontró modelo entrenado. Usando Transformer sin entrenar.")
            procesador.comparar_todos(ruta_imagen)
        elif opcion == "4":
            print("\n Cargando modelo y demostrando...")
            ruta_modelo = input("Ruta del modelo (.pkl): ").strip()
            ruta_imagen = input("Ruta de imagen (Enter para sintética): ").strip() or None
            try:
                procesador.transformador.cargar_modelo(ruta_modelo)
                procesador.demostrar_mejora_imagen(ruta_imagen)
            except FileNotFoundError:
                print(f" No se encontró el modelo en: {ruta_modelo}")
        else:
            print(" Opción no válida")
    except KeyboardInterrupt:
        print("\n\n  Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n Error durante la ejecución: {e}")
    print("\n Programa finalizado")
    print(" Archivos generados:")
    print("  - transformer_mejorado.pkl: Modelo entrenado")
    print("  - entrenamiento_progreso.png: Gráficos de entrenamiento") 
    print("  - mejora_imagenes_transformer.png: Resultados de mejora")
    print("  - comparacion_completa.png: Comparación de métodos")
if __name__ == "__main__":
    principal()