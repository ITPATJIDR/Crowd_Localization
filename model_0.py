from ultralytics import YOLO

def main():
    model = YOLO('yolov8x.pt')
    results = model.train(data='./head.v1i.yolo8/data.yaml', epochs=100)

    print(results)


if __name__ == '__main__':
    main()

