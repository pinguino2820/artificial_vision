from onvif import ONVIFCamera
import time 

# CONFIGURACIÓN DE LA CÁMARA

IP_CAMARA = "192.168.1.50" 	# Cambia por la IP local de tu H8c
PUERTO_ONVIF = 80 		 	# Puerto ONVIF estándar para EZVIZ/Hikvision
USUARIO = "admin"          	# Por defecto en ONVIF local siempre es admin
CONTRASENA = "ABCDEF" 		# Reemplaza con el Verification Code (6 letras) 

try: 
    # 1. Inicializar la conexión ONVIF
    print(f"[+] Conectando a la cámara en {IP_CAMARA}:{PUERTO_ONVIF}...")
    camara = ONVIFCamera(IP_CAMARA, PUERTO_ONVIF, USUARIO, CONTRASENA) 

    #2. Crear los servicios necesarios
    media_service = camara.create_media_service()
    ptz_service = camara.create_ptz_service() 

    # 3. Obtener el perfil de transmisión activo 
    # (necesario para direccionar el comando)
    perfiles = media_service.GetProfiles() 
    token_perfil = perfiles[0].token 
    print(f"[+] Conexión exitosa. Token de perfil obtenido: {token_perfil}")

except Exception as e:
      print(f"[-] Error al conectar con la cámara: {e}")
      exit(1) 

# FUNCIONES DE MOVIMIENTO (PTZ)

def mover_camara(velocidad_x, velocidad_y, duracion_segundos):
    """
    Mueve la cámara de forma continua en una dirección y se detiene tras N segundos.
    Valores para velocidad_x (Paneo / Horizontal): -1.0 (Izquierda) a 1.0 (Derecha)
    Valores para velocidad_y (Cabeceo / Vertical): -1.0 (Abajo) a 1.0 (Arriba)
    """
    try:
        print(f"[*] Moviendo cámara (X: {velocidad_x}, Y: {velocidad_y}) por {duracion_segundos}s...") 

        #Estructura del comando de movimiento continuo ONVIF
        solicitud_movimiento = ptz_service.create_type('ContinuousMove')
        solicitud_movimiento.ProfileToken = token_perfil
        solicitud_movimiento.Velocity = {
                'PanTilt': {
                'x': velocidad_x,
                 'y': velocidad_y
            }
        } 

        # Iniciar el movimiento
        ptz_service.ContinuousMove(solicitud_movimiento) 
        # Esperar el tiempo indicado mientras la cámara gira
        time.sleep(duracion_segundos) 
        # Detener el movimiento de inmediato
        solicitud_parada = ptz_service.create_type('Stop') 
        solicitud_parada.ProfileToken = token_perfil 
        solicitud_parada.PanTilt = True 
        ptz_service.Stop(solicitud_parada) 
        print("[+] Movimiento detenido.")
    
    except Exception as e:
        print(f"[-] Error durante el movimiento: {e}") 
        
# PRUEBA DE FUNCIONAMIENTO
if __name__ == "main": 
    # Ejemplo 1: Girar a la izquierda a velocidad media durante 1.5 segundos
    mover_camara(velocidad_x=-0.5, velocidad_y=0.0, duracion_segundos=1.5) 
    time.sleep(2) # Pausa de estabilidad
     
    # Ejemplo 2: Inclinar la cámara hacia arriba durante 1 segundo
    mover_camara(velocidad_x=0.0, velocidad_y=0.4, duracion_segundos=1.0) 

