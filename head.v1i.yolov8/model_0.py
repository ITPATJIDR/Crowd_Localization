from ultralytics import YOLO

def main():
    model = YOLO('yolov8x.pt')
    results = model.train(data='data.yaml', epochs=1)

    print(results)


if __name__ == '__main__':
    main()

