'''
Программа распознает 6 типов автомобилей на видео, производит подсчет каждого из них, а также
подсчет общего числа атомобилей на данной кадре и заносит все резльтаты в .json файл
'''
import base64
from typing import OrderedDict

# RUN THE CODE:
# python car_counter_yolov3_6_classes.py -y yolo --input videos/traffic.mp4 --output output --skip-frames 5

# импортируем необходимые библиотеки и функции
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import imutils
import dlib
import json
import cv2
import os
from matplotlib import pyplot as plt

from DetectionRectangle import DetectionRectangle
from VehicleType import VehicleType

from dotenv import load_dotenv
import struct

load_dotenv()


# парсер аргументов с командной строки
class CarCounter:

    def __init__(self, yolo: str, net_input_dir: str, output_dir: str, skip_frames: int,
                 detection_rectangles: list[DetectionRectangle], video: str, confidence_lower_bound=0.90, DEBUG=False):
        self.yolo = yolo
        self.net_input = net_input_dir
        self.output = output_dir
        self.skip_frames = skip_frames
        self.detection_rectangles = detection_rectangles
        self.confidence_lower_bound = confidence_lower_bound
        self.video = video
        self.lower_relevance_margin = 100
        self.upper_relevance_margin = 100
        self.left_relevance_margin = 100
        self.right_relevance_margin = 100
        self.video_path = 'video_path'
        self.DEBUG = DEBUG

    def draw_detection_areas(self, detection_frame, detection_areas: list[DetectionRectangle]):
        for detection_area in detection_areas:
            cv2.rectangle(detection_frame,
                          (detection_area.lowerLeft.first, detection_area.lowerLeft.second),
                          (detection_area.upperRight.first, detection_area.upperRight.second),
                          (0, 0, 255), 4)

    def is_overlapping(self, rectangles, new_rectangle):
        for rectangle in rectangles:
            if not ((rectangle[0] >= new_rectangle[2]) or (rectangle[2] <= new_rectangle[0])
                    or (rectangle[3] <= new_rectangle[1]) or (rectangle[1] >= new_rectangle[3])):
                return True
        return False

    def is_point_inside_rectangle(self, point, rectangle):
        return rectangle[0] < point[0] < rectangle[2] and rectangle[1] < point[1] < rectangle[3]

    # Функция считает общее количество объектов класса, появившихся на видео
    def count_objects(self, vehicles: OrderedDict, detection_rectangles: list[DetectionRectangle],
                      vehicle_type: VehicleType):
        for vehicle_id, vehicle_centroid in vehicles.items():
            for detection_rectangle in detection_rectangles:
                if self.is_point_inside_rectangle(vehicle_centroid,
                                                  [detection_rectangle.lowerLeft.first,
                                                   detection_rectangle.lowerLeft.second,
                                                   detection_rectangle.upperRight.first,
                                                   detection_rectangle.upperRight.second]):

                    if vehicle_type == VehicleType.CAR:
                        detection_rectangle.detected_car_ids.add(vehicle_id)
                    elif vehicle_type == VehicleType.BUS:
                        detection_rectangle.detected_bus_ids.add(vehicle_id)
                    else:
                        raise Exception("unsupported vehicle type")

    # Функция рисует ID-шники и центроиды объектов
    def draw_centroids(self, frame, objects, trackableObjects):
        # анализируем массив отслеживаемых объектов
        for (objectID, centroid) in objects.items():

            # проверяем существует ли отслеживаемый объект для данного ID
            to = trackableObjects.get(objectID, None)

            # если его нет, то создаем новый, соответствующий данному центроиду
            if to is None:
                to = TrackableObject(objectID, centroid)

            # в любом случае помещаем объект в словарь
            # (1) ID (2) объект
            trackableObjects[objectID] = to

            # изобразим центроид и ID объекта на кадре
            text = "ID {}".format(objectID + 1)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            cv2.circle(frame, (10, 10), 4, (0, 255, 0), -1)

    def create_detections_rectangle_dict(self, detection_rectangles: list[DetectionRectangle]) -> dict[
        int, DetectionRectangle]:
        detection_rectangles_dict = {}
        for idx, detection_rectangle in enumerate(detection_rectangles):
            detection_rectangles_dict[idx] = detection_rectangle
        return detection_rectangles_dict

    def get_relevant_video_boundary(self, frame):
        if len(self.detection_rectangles) == 0:
            raise Exception("CR should have at least one detection rectangle")

        for detection_rectangle in self.detection_rectangles:
            detection_rectangle.lowerLeft.second = frame.shape[0] - detection_rectangle.lowerLeft.second
            detection_rectangle.upperRight.second = frame.shape[0] - detection_rectangle.upperRight.second

        lower_left_boundary_corner = [self.detection_rectangles[0].lowerLeft.first,
                                      self.detection_rectangles[0].lowerLeft.second]
        upper_right_boundary_corner = [self.detection_rectangles[0].upperRight.first,
                                       self.detection_rectangles[0].upperRight.second]

        for detection_rectangle in self.detection_rectangles:
            lower_left_boundary_corner[0] = min(detection_rectangle.lowerLeft.first,
                                                lower_left_boundary_corner[0])
            lower_left_boundary_corner[1] = min(detection_rectangle.lowerLeft.second,
                                                lower_left_boundary_corner[1])

            upper_right_boundary_corner[0] = max(detection_rectangle.upperRight.first,
                                                 upper_right_boundary_corner[0])
            upper_right_boundary_corner[1] = max(detection_rectangle.upperRight.second,
                                                 upper_right_boundary_corner[1])

        lower_left_boundary_corner[0] = max(0, lower_left_boundary_corner[0] - self.lower_relevance_margin)
        lower_left_boundary_corner[1] = max(0, lower_left_boundary_corner[1] - self.left_relevance_margin)

        upper_right_boundary_corner[0] += self.upper_relevance_margin
        upper_right_boundary_corner[1] += self.right_relevance_margin

        return lower_left_boundary_corner, upper_right_boundary_corner

    def shift_detection_rectangles(self, lower_left_corner: list[int, int]):
        for detection_rectangle in self.detection_rectangles:
            detection_rectangle.lowerLeft.first -= lower_left_corner[0]
            detection_rectangle.lowerLeft.second -= lower_left_corner[1]

            detection_rectangle.upperRight.first -= lower_left_corner[0]
            detection_rectangle.upperRight.second -= lower_left_corner[1]

    def save_video(self):
        video_path = os.getenv(self.video_path) + self.net_input
        if os.path.isfile(video_path):
            os.remove(video_path)

        with open(video_path, "wb") as out_file:  # open for [w]riting as [b]inary
            out_file.write(base64.b64decode(self.video))

    def run(self):

        net = cv2.dnn.readNet(self.yolo + "/yolov3.weights", self.yolo + "/yolov3_608.cfg")
        # классы объектов, которые могут быть распознаны алгоритмом
        with open(self.yolo + "/yolov3_608.names", 'r') as f:
            CLASSES = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        # Размеры входного изображения
        inpWidth = 608
        inpHeight = 608
        self.save_video()
        vs = cv2.VideoCapture(os.getenv('video_path') + self.net_input)

        # объявляем инструмент для записи конечного видео в файл, указываем путь
        writer = None
        output_count = 1
        while True:
            # если в директории вывода уже больше 20 файлов, то она очищается
            if output_count > 20:
                for file in os.listdir(self.output):
                    os.remove(os.getcwd() + "/output/" + file)
                    output_count = 1

            if "{}_proccesed.avi".format(output_count) not in os.listdir("../" + self.output):
                writer_path = self.output + "/{}_proccesed.avi".format(output_count)
                break
            else:
                output_count += 1

        # инициализируем размеры кадра как пустые значения
        # они будут переназначены при анализе первого кадра и только
        # это ускорит работу программы
        width = None
        height = None

        # инициализируем алгоритм трекинга
        # maxDisappeared = кол-во кадров, на которое объект может исчезнуть с видео и потом опять
        # будет распознан
        # maxDistance = максимальное расстояние между центрами окружностей, вписанных в боксы машин
        # Если расстояние меньше заданного, то происходит переприсваение ID
        car_ct = CentroidTracker()
        car_ct.maxDisappeared = 10
        person_ct = CentroidTracker()
        person_ct.maxDisappeared = 10
        truck_ct = CentroidTracker()
        truck_ct.maxDisappeared = 10
        bike_ct = CentroidTracker()
        bike_ct.maxDisappeared = 10
        bicycle_ct = CentroidTracker()
        bicycle_ct.maxDisappeared = 10
        bus_ct = CentroidTracker()
        bus_ct.maxDisappeared = 10

        # сам список трекеров
        trackers = []
        # список объектов для трекинга
        car_trackableObjects = {}
        bus_trackableObjects = {}

        # person_trackableObjects = {}
        # truck_trackableObjects = {}
        # bike_trackableObjects = {}
        # bicycle_trackableObjects = {}

        # глобальные переменные для счетчиков
        total_cars, temp_cars = 0, None
        total_persons, temp_persons = 0, None
        total_trucks, temp_trucks = 0, None
        total_buses, temp_buses = 0, None
        total_bikes, temp_bikes = 0, None
        total_bicycles, temp_bicycles = 0, None

        # полное число кадров в видео
        totalFrames = 0

        total = 0

        # статус: распознавание или отслеживание
        status = None

        # номер кадра видео
        frame_number = 0

        # инициализируем нулями
        count_cars, count_persons, count_trucks, count_buses, count_bikes, count_bicycles = 0, 0, 0, 0, 0, 0

        data = {
            "cars": str(count_cars),
            "people": str(count_persons),
            "trucks": str(count_trucks),
            "buses:": str(count_buses),
            "motorcycles:": str(count_bikes),
            "bycicles": str(count_bicycles),
        }

        lower_left_corner, upper_right_corner = self.get_relevant_video_boundary(vs.read()[1])
        self.shift_detection_rectangles(lower_left_corner)

        detection_rectangles_dict = self.create_detections_rectangle_dict(self.detection_rectangles)

        # проходим через каждый кадр видео
        while True:
            frame_number += 1
            frame = vs.read()
            frame = frame[1]

            # если кадр является пустым значением, значит был достигнут конец видео
            if frame is None:
                break

            # изменим размер кадра для ускорения работы
            frame = imutils.resize(frame)
            frame = frame[lower_left_corner[1]: upper_right_corner[1], lower_left_corner[0]: upper_right_corner[0], :]

            self.draw_detection_areas(frame, self.detection_rectangles)

            # для работы библиотеки dlib необходимо изменить цвета на RGB вместо BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # размеры кадра
            if width is None or height is None:
                height, width, channels = frame.shape

            # этот список боксов может быть заполнен двумя способами:
            # (1) детектором объектов
            # (2) трекером наложений из библиотеки dlib
            # 2 автомобили
            # 0 человек
            # 7 грузовки
            # 5 автобус
            # 3 мото
            # 1 велосипед
            car_rects = []
            person_rects = []
            truck_rects = []
            bus_rects = []
            bike_rects = []
            bicycle_rects = []

            # задаем путь записи конечного видео
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(writer_path, fourcc, 30,
                                         (width, height), True)

            # каждые N кадров (указанных в аргументе "skip_frames" производится ДЕТЕКТРОВАНИЕ машин
            # после этого идет ОТСЛЕЖИВАНИЕ их боксов
            # это увеличивает скорость работы программы
            if totalFrames % self.skip_frames == 0:
                # создаем пустой список трекеров
                trackers = []
                # список номером классов (нужен для подписи класса у боксов машин
                class_ids = []
                # сколь машин на данном кадре
                count = 0

                # already detected cars
                car_rectangles = []

                status = "Detecting..."

                # получаем blob-модель из кадра и пропускаем ее через сеть, чтобы получить боксы распознанных объектов
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                # анализируем список боксов
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)

                        if class_id == 0:  # если обнаружена "background" - пропускаем
                            pass

                        confidence = scores[class_id]
                        # получаем ID наиболее "вероятных" объектов
                        if confidence > self.confidence_lower_bound:
                            # находятся координаты центроида бокса
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            # это ИМЕННО ШИРИНА - то есть расстояние от левого края до правого
                            w = int(detection[2] * width)
                            # это ИМЕННО ВЫСОТА - то есть расстояние от верхнего края до нижнего
                            h = int(detection[3] * height)

                            # Координаты бокса (2 точки углов)
                            x1 = int(center_x - w / 2)
                            y1 = int(center_y - h / 2)
                            x2 = x1 + w
                            y2 = y1 + h

                            if not self.is_overlapping(car_rectangles, (x1, y1, x2, y2)):
                                car_rectangles.append((x1, y1, x2, y2))

                                # возьмем максимальный радиус для CentroidTracker пропорционально размеру машины
                                person_ct.maxDistance = w
                                bike_ct.maxDistance = w
                                bicycle_ct.maxDistance = w
                                bus_ct.maxDistance = w
                                truck_ct.maxDistance = w
                                car_ct.maxDistance = w

                                count += 1

                                # рисую бокс для теста
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                                cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)

                                # создаем трекер ДЛЯ КАЖДОЙ МАШИНЫ
                                tracker = dlib.correlation_tracker()
                                # создаем прямоугольник из бокса (фактически, это и есть бокс)
                                rect = dlib.rectangle(x1, y1, x2, y2)
                                # трекер начинает отслеживание КАЖДОГО БОКСА
                                tracker.start_track(rgb, rect)
                                # и каждый трекер помещается в общий массив
                                trackers.append(tracker)
                                class_ids.append(class_id)



            # если же кадр не явялется N-ым, то необходимо работать с массивом сформированных ранее трекеров, а не боксов
            else:
                for tracker, class_id in zip(trackers, class_ids):
                    status = "Tracking..."

                    '''
                    На одном кадре машина была распознана. Были получены координаты ее бокса. ВСЕ последующие 5 кадров эти координаты
                    не обращаются в нули, а изменяются благодяра update(). И каждый их этих пяти кадров в rects помещается предсказанное
                    программой местоположение бокса!
                    '''
                    tracker.update(rgb)
                    # получаем позицию трекера в списке(это 4 координаты)
                    pos = tracker.get_position()

                    # из трекера получаем координаты бокса, соответствующие ему
                    x1 = int(pos.left())
                    y1 = int(pos.top())
                    x2 = int(pos.right())
                    y2 = int(pos.bottom())

                    # рисую бокс
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    obj_class = CLASSES[class_id]
                    # 2 автомобили
                    # 0 человек
                    # 7 грузовки
                    # 5 автобус
                    # 3 мото
                    # 1 велосипед

                    if obj_class == "car":
                        car_rects.append((x1, y1, x2, y2))
                    elif obj_class == "bus":
                        bus_rects.append((x1, y1, x2, y2))

                    # elif obj_class == "person":
                    #     person_rects.append((x1, y1, x2, y2))
                    # elif obj_class == "truck":
                    #     truck_rects.append((x1, y1, x2, y2))
                    # elif obj_class == "motorcycle":
                    #     bike_rects.append((x1, y1, x2, y2))
                    # elif obj_class == "bicycle":
                    #     bicycle_rects.append((x1, y1, x2, y2))

            '''
            После детекта первой машины и до конца работы программы rects больше никогда не станут []. 
            Единственное условие, при котором len(objects.keys()) станет равно 0. Это если истичет предел maxDisappeared, то есть
            rects так и будут НЕпустым массивом, но машина слишком надолго исчезнет из виду.
            '''
            cars = car_ct.update(car_rects)
            buses = bus_ct.update(bus_rects)

            # persons = person_ct.update(person_rects)
            # trucks = truck_ct.update(truck_rects)
            # bikes = bike_ct.update(bike_rects)
            # bicycles = bicycle_ct.update(bicycle_rects)

            if cars != {}:
                self.count_objects(cars, self.detection_rectangles, VehicleType.CAR)
            if buses != {}:
                self.count_objects(buses, self.detection_rectangles, VehicleType.BUS)

            # Данные для вывода на экран
            info = [

                       ("cars in line {}: ".format(i + 1), len(detection_rectangles_dict[i].detected_car_ids)) for i in
                       range(len(self.detection_rectangles))
                   ] + [("buses in line {}: ".format(i + 1), len(detection_rectangles_dict[i].detected_bus_ids)) for i
                        in range(len(self.detection_rectangles))]

            # данные для записи в JSON
            data = {
                "cars line 1": str(count_cars),
                "cars line 2": str(count_persons),
                "cars line 3": str(count_trucks),
                "cars line 4:": str(count_buses),
                "cars line 5:": str(count_bikes),
                "buses line 1: ": str(count_cars),
                "buses line 2: ": str(count_persons),
                "buses line 3: ": str(count_trucks),
                "buses line 4: ": str(count_buses),
                "buses line 5: ": str(count_bikes),
            }

            self.draw_centroids(frame, cars, car_trackableObjects)
            self.draw_centroids(frame, buses, bus_trackableObjects)

            # изобразим информаци о количестве машин на краю кадра
            for (i, (object_class, total)) in enumerate(info):
                text = "{}: {}".format(object_class, total)
                cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)

            cv2.putText(frame, "Now: " + str(count), (width - 120, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0),
                        2)

            # записываем конечный кадр в указанную директорию
            if writer is not None:
                writer.write(frame)

            if self.DEBUG:
                # показываем конечный кадр в отдельном окне
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # для прекращения работы необходимо нажать клавишу "q"
                if key == ord("q"):
                    break

            # т.к. все выше-обработка одного кадра, то теперь необходимо увеличить количесвто кадров
            # и обновить счетчик
            totalFrames += 1

        if self.DEBUG:
            # график выводится на экран в конце работы программы        
            plt.show()

        # записываю все полученные данные в json файл

        with open("../" + self.output + "/" + "analysis_results_{}.json".format(output_count), 'w') as f:
            json.dump(data, f)

        # закрываем все окна
        cv2.destroyAllWindows()

        info = []
        for i in range(len(self.detection_rectangles)):
            info.append(dict(id=i, detectedCars=len(detection_rectangles_dict[i].detected_car_ids),
                             detectedBuses=len(detection_rectangles_dict[i].detected_bus_ids),
                             connectionId=detection_rectangles_dict[i].connectionId))

        return str(info)
