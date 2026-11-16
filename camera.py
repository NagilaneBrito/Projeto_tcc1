
import cv2
import supervision as sv
from ultralytics import YOLO
from urllib.parse import quote
import typer

# Carregar modelo YOLO
model = YOLO("best.pt")
class_names = model.names

print("Class names:", class_names)

app = typer.Typer(help="YOLO detection on Intelbras IP camera")


def process(source: str, output_file: str, conf_threshold: float = 0.25):

    # Conectar a camera Intelbras
    camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    # Reduzir buffer para tentar diminuir atraso
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("Nao foi possivel conectar a camera Intelbras.")
        return

    print("Camera Intelbras conectada com sucesso!")

    largura = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FPS utilizado na gravacao
    fps = 15

    # Criar arquivo de video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_file,
        fourcc,
        fps,
        (640, 360)
    )

    if not out.isOpened():
        print("Nao foi possivel criar o arquivo de video.")
        camera.release()
        return

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Falha ao receber imagem da camera.")
            break

        # Reduzir resolucao para melhorar velocidade do YOLO
        frame = cv2.resize(
            frame,
            (640, 360),
            interpolation=cv2.INTER_AREA
        )

        # Executar YOLO
        results = model(
            frame,
            conf=conf_threshold,
            imgsz=320,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Desenhar bounding boxes
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        # Criar rotulos
        labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(
                detections.class_id,
                detections.confidence
            )
        ]

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # Gravar video com deteccoes
        out.write(annotated_frame)

        # Mostrar resultado
        cv2.imshow("YOLO - Camera Intelbras", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Gravacao salva em: {output_file}")


@app.command()
def webcam(
    source: str = typer.Option(
        ...,
        "--source",
        help="Endereco RTSP da camera Intelbras"
    ),
    output_file: str = typer.Option(
        "gravacao_intelbras.mp4",
        "--output-file",
        "-o",
        help="Nome do arquivo de saida"
    ),
    conf: float = typer.Option(
        0.25,
        "--conf",
        "-c",
        help="Confidence threshold"
    ),
):

    process(source, output_file, conf)


if __name__ == "__main__":
    app()
