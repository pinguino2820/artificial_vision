
from pyezviz import EzvizClient, EzvizCamera
from onvif import ONVIFCamera
import time

# CONFIGURACIÓN GENERAL

# 1. Credenciales de la App (Nube)
EMAIL_EZVIZ = "tu_correo@gmail.com"
PASSWORD_EZVIZ = "TuContrasenaDeLaApp"
REGION = "la"  # "la" para Latinoamérica 

# 2. Credenciales del dispositivo (Físicas)
SERIAL_CAMARA = "S12345678"        # Impreso en la etiqueta (9 dígitos)
VERIFICATION_CODE = "ABCDEF"      # Código de 6 letras en mayúsculas (etiqueta) 

# Variables globales para los servicios
camara_cloud = None
ptz_local = None
token_perfil_local = None

def inicializar_sistema_hibrido():
    global camara_cloud, ptz_local, token_perfil_local

    try:
        # STEP 1: Conectar a la Nube para leer el estado del ecosistema
        print("[+] 1. Conectando a los servidores Cloud de EZVIZ...")
        cliente_cloud = EzvizClient(EMAIL_EZVIZ, PASSWORD_EZVIZ, REGION)
        cliente_cloud.login()

        camara_cloud = EzvizCamera(cliente_cloud, SERIAL_CAMARA)
        estado_cloud = camara_cloud.status()

        # Obtener dinámicamente la IP local que la nube reporta para la cámara
        ip_local = estado_cloud.get('local_ip')
        print(f"[+] Cámara detectada en la nube. IP Local reportada: {ip_local}")

        if not ip_local or ip_local == "0.0.0.0":
            raise Exception("No se pudo determinar la IP local de la cámara. ¿Está conectada al mismo Wi-Fi?")

        # STEP 2: Inicializar la conexión ONVIF local usando la IP obtenida de la nube
        print(f"[+] 2. Abriendo canal ONVIF local de alta velocidad en {ip_local}...")
        camara_local = ONVIFCamera(ip_local, 80, 'admin', VERIFICATION_CODE)

        media_service = camara_local.create_media_service()
        ptz_local = camara_local.create_ptz_service()

        perfiles = media_service.GetProfiles()
        token_perfil_local = perfiles.token
        print("[+] Sistema híbrido listo. Control de movimiento local sincronizado.")
        return True

    except Exception as e:
        print(f"[-] Error en la inicialización: {e}")
        return False

# ACCIONES REPLICADAS
def mover_local(vel_x, vel_y, duracion):
    # Mueve la cámara localmente de forma inmediata (Cero Latencia)
    if not ptz_local: return
    try:
        print(f"[*] [LOCAL] Moviendo PTZ (X: {vel_x}, Y: {vel_y})")
        solicitud = ptz_local.create_type('ContinuousMove')
        solicitud.ProfileToken = token_perfil_local
        solicitud.Velocity = {'PanTilt': {'x': vel_x, 'y': vel_y}}

        ptz_local.ContinuousMove(solicitud)
        time.sleep(duracion)
        # Parada inmediata
        parada = ptz_local.create_type('Stop')
        parada.ProfileToken = token_perfil_local
        parada.PanTilt = True
        ptz_local.Stop(parada)

    except Exception as e:
        print(f"[-] Error en movimiento local: {e}")

def alternar_privacidad_cloud(activar: bool):
    # Cambia configuraciones avanzadas usando la API Cloud.
    if not camara_cloud: return
    estado = 1 if activar else 0
    accion = "Activando" if activar else "Desactivando"
    print(f"[*] [CLOUD] {accion} modo privacidad en la cámara...")
    try:
        camara_cloud.switch("privacy", estado)
    except Exception as e:
        print(f"[-] Error al cambiar modo privacidad en la nube: {e}")

# FLUJO DE EJECUCIÓN

if __name__ == "main":
    if inicializar_sistema_hibrido():
        # 1. Asegurarnos a través de la nube de que la cámara no esté en modo "Sleep"
        alternar_privacidad_cloud(activar=False)
        time.sleep(2) # Esperar a que despierte 

        # 2. Ejecutar movimientos ultra-rápidos mediante la red LAN.
        print("\n--- Iniciando Patrullaje de Prueba Local ---")
        mover_local(vel_x=0.6, vel_y=0.0, duracion=1.5)   # Girar rápido a la derecha
        time.sleep(1)
        mover_local(vel_x=-0.6, vel_y=0.0, duracion=1.5)  # Regresar a la izquierda 

        print("\n--- Patrullaje finalizado ---")

        # 3. Al terminar nuestras tareas de monitoreo, la mandamos a dormir vía Cloud.
        alternar_privacidad_cloud(activar=True)