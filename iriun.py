
import cv2

numero = 0  

camera = cv2.VideoCapture(numero)

if not camera.isOpened():
    print("Não foi possível abrir essa câmera.")
else:
    while True:
        sucesso, imagem = camera.read()

        if not sucesso:
            print("Não foi possível receber a imagem.")
            break

        cv2.imshow("Teste da camera", imagem)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
