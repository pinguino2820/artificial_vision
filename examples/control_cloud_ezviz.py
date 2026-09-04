
import time
from pyezviz import EzvizClient, EzvizCamera

# CREDENCIALES DE TU CUENTA EZVIZ (Las mismas de la App móvil)

EMAIL_O_USUARIO = "tu_correo@gmail.com"
CONTRASENA = "TuContrasenaDeLaApp"
REGION = "la"                      # "la" para Latinoamérica (o "eu", "us" según tu cuenta)
SERIAL_CAMARA = "S12345678"        # El número de serie de 9 dígitos de tu H8c 

try:
    # 1. Crear el cliente e iniciar sesión en la nube de EZVIZ      
    print("[+] Conectando con los servidores de EZVIZ...")
    cliente = EzvizClient(EMAIL_O_USUARIO, CONTRASENA, REGION)
    cliente.login()
    print("[+] Autenticación exitosa.")

    # 2. Cargar la instancia de tu cámara H8c mediante su número de serie
    # Nota: También puedes usar cliente.load_cameras() para listar todas tus cámaras        
    camara = EzvizCamera(cliente, SERIAL_CAMARA)
    print(f"[+] Conectado a la cámara: {camara.status().get('name', 'H8c')}")

except Exception as e:
    print(f"[-] Error de autenticación o conexión: {e}")
    exit(1)

# INTERACCIÓN Y COMANDOS DE CONTROL
def probar_funciones_cloud():
    try:
        # Ejemplo 1: Mover la cámara usando comandos de dirección.
        # Direcciones soportadas: 'up', 'down', 'left', 'right'
        # Velocidades soportadas: De 1 (lento) a 5 (rápido)

        print("[*] Girando la cámara a la derecha...")
        camara.move(direction="right", speed=3)

        # Esperamos unos segundos y realizamos otro movimiento.
        time.sleep(3)
        print("[*] Girando la cámara hacia arriba...")
        camara.move(direction="up", speed=3)

        time.sleep(2)
        # Ejemplo 2: Cambiar configuraciones que no se pueden por ONVIF local.
        # Activar el modo privacidad (la cámara físicamente se oculta/duerme)
        print("[*] Activando modo privacidad (Sleep)...")
        camara.switch("privacy", 1)  # 1 = Activar, 0 = Desactivar 

        time.sleep(5)
        # Desactivar el modo privacidad para despertar la cámara.
        print("[*] Desactivando modo privacidad...")
        camara.switch("privacy", 0)

        # Ejemplo 3: Consultar el estado actual en tiempo real de la nube.
        estado = camara.status()
        print("\n--- ESTADO DE LA CÁMARA ---")
        print(f"IP Local: {estado.get('local_ip')}")
        print(f"Versión de firmware: {estado.get('version')}")
        print(f"Notificaciones de alarma activas: {estado.get('alarm_notify')}")

    except Exception as e:
        print(f"[-] Error al ejecutar comandos en la nube: {e}")

if __name__ == "main":
    probar_funciones_cloud()