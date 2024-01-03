import base64
from typing import OrderedDict

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

load_dotenv()


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
        return rectangle[0] < point[0] < rectangle[2] and rectangle[1] > point[1] > rectangle[3]

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

    def draw_centroids(self, frame, objects, trackableObjects):
        for (objectID, centroid) in objects.items():

            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            trackableObjects[objectID] = to

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
        with open(self.yolo + "/yolov3_608.names", 'r') as f:
            CLASSES = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        inpWidth = 608
        inpHeight = 608
        self.save_video()
        vs = cv2.VideoCapture(os.getenv('video_path') + self.net_input)

        writer = None
        output_count = 1
        while True:
            if output_count > 20:
                for file in os.listdir(self.output):
                    os.remove(os.getcwd() + "/output/" + file)
                    output_count = 1

            if "{}_proccesed.avi".format(output_count) not in os.listdir("../" + self.output):
                writer_path = self.output + "/{}_proccesed.avi".format(output_count)
                break
            else:
                output_count += 1

        width = None
        height = None

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

        trackers = []
        car_trackableObjects = {}
        bus_trackableObjects = {}

        total_cars, temp_cars = 0, None
        total_persons, temp_persons = 0, None
        total_trucks, temp_trucks = 0, None
        total_buses, temp_buses = 0, None
        total_bikes, temp_bikes = 0, None
        total_bicycles, temp_bicycles = 0, None

        totalFrames = 0

        total = 0

        status = None

        frame_number = 0

        count_cars, count_persons, count_trucks, count_buses, count_bikes, count_bicycles = 0, 0, 0, 0, 0, 0

        data = {
            "cars": str(count_cars),
            "people": str(count_persons),
            "trucks": str(count_trucks),
            "buses:": str(count_buses),
            "motorcycles:": str(count_bikes),
            "bycicles": str(count_bicycles),
        }

        detection_rectangles_dict = self.create_detections_rectangle_dict(self.detection_rectangles)

        while True:
            frame_number += 1
            frame = vs.read()
            frame = frame[1]

            if frame is None:
                break

            frame = imutils.resize(frame)

            self.draw_detection_areas(frame, self.detection_rectangles)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if width is None or height is None:
                height, width, channels = frame.shape

            car_rects = []
            person_rects = []
            truck_rects = []
            bus_rects = []
            bike_rects = []
            bicycle_rects = []

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(writer_path, fourcc, 30,
                                         (width, height), True)

            if totalFrames % self.skip_frames == 0:
                trackers = []
                class_ids = []
                count = 0

                car_rectangles = []

                status = "Detecting..."

                blob = cv2.dnn.blobFromImage(frame, 0.00392, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)

                        if class_id == 0:
                            pass

                        confidence = scores[class_id]
                        if confidence > self.confidence_lower_bound:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x1 = int(center_x - w / 2)
                            y1 = int(center_y - h / 2)
                            x2 = x1 + w
                            y2 = y1 + h

                            if not self.is_overlapping(car_rectangles, (x1, y1, x2, y2)):
                                car_rectangles.append((x1, y1, x2, y2))

                                person_ct.maxDistance = w
                                bike_ct.maxDistance = w
                                bicycle_ct.maxDistance = w
                                bus_ct.maxDistance = w
                                truck_ct.maxDistance = w
                                car_ct.maxDistance = w

                                count += 1

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                                cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)

                                tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(x1, y1, x2, y2)
                                tracker.start_track(rgb, rect)
                                trackers.append(tracker)
                                class_ids.append(class_id)



            else:
                for tracker, class_id in zip(trackers, class_ids):
                    status = "Tracking..."

                    tracker.update(rgb)
                    pos = tracker.get_position()

                    x1 = int(pos.left())
                    y1 = int(pos.top())
                    x2 = int(pos.right())
                    y2 = int(pos.bottom())

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    obj_class = CLASSES[class_id]

                    if obj_class == "car":
                        car_rects.append((x1, y1, x2, y2))
                    elif obj_class == "bus":
                        bus_rects.append((x1, y1, x2, y2))

            cars = car_ct.update(car_rects)
            buses = bus_ct.update(bus_rects)

            if cars != {}:
                self.count_objects(cars, self.detection_rectangles, VehicleType.CAR)
            if buses != {}:
                self.count_objects(buses, self.detection_rectangles, VehicleType.BUS)

            info = [

                       ("cars in line {}: ".format(i + 1), len(detection_rectangles_dict[i].detected_car_ids)) for i in
                       range(len(self.detection_rectangles))
                   ] + [("buses in line {}: ".format(i + 1), len(detection_rectangles_dict[i].detected_bus_ids)) for i
                        in range(len(self.detection_rectangles))]

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

            for (i, (object_class, total)) in enumerate(info):
                text = "{}: {}".format(object_class, total)
                cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)

            cv2.putText(frame, "Now: " + str(count), (width - 120, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0),
                        2)

            if writer is not None:
                writer.write(frame)

            if self.DEBUG:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            totalFrames += 1

        if self.DEBUG:
            plt.show()

        with open("../" + self.output + "/" + "analysis_results_{}.json".format(output_count), 'w') as f:
            json.dump(data, f)

        cv2.destroyAllWindows()

        info = []
        for i in range(len(self.detection_rectangles)):
            info.append(dict(id=i, detectedCars=len(detection_rectangles_dict[i].detected_car_ids),
                             detectedBuses=len(detection_rectangles_dict[i].detected_bus_ids),
                             connectionId=detection_rectangles_dict[i].connectionId))

        return str(info)
