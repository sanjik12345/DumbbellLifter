import cv2
import mediapipe as mp
import numpy as np

# Инициализация объектов для рисования и позовых моделей из библиотеки MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Открытие захвата видео с веб-камеры
cap = cv2.VideoCapture(0)

# Переменные для счётчика подъемов и этапа упражнения
counter = 0  # Счетчик повторений
stage = None  # Текущий этап (вверх или вниз)

def calculate_angle(a, b, c):
    a = np.array(a)  # Точка A
    b = np.array(b)  # Точка B (центр угла)
    c = np.array(c)  # Точка C

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Настройка экземпляра позовой модели MediaPipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()  # Чтение кадра из видеопотока

        # Переключение цветового пространства кадра с BGR на RGB для работы с MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Оптимизация: запрещаем запись в image

        # Обработка кадра для определения позы
        results = pose.process(image)

        # Возвращаем цветовое пространство с RGB на BGR для отображения с помощью OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Извлечение ключевых точек позы
        try:
            landmarks = results.pose_landmarks.landmark

            # Получение координат плеча, локтя и запястья для левой руки
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Вычисление угла между плечом, локтем и запястьем
            angle = calculate_angle(shoulder, elbow, wrist)

            # Отображение угла на экране рядом с локтем
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Логика счётчика упражнений
            if angle > 160:
                stage = "down"  # Фаза опускания
            if angle < 30 and stage == 'down':
                stage = "up"  # Фаза подъема
                counter += 1  # Увеличение счетчика повторений
                print(counter)

        except:
            pass  # Пропуск ошибок, если landmarks не найдены

        # Отображение данных счётчика на экране
        # Рисуем прямоугольник для счетчика и состояния
        cv2.rectangle(image, (0, 0), (270, 78), (245, 117, 16), -1)

        # Текст для количества повторений
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Текст для текущей фазы
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (90, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Отрисовка позы с использованием MediaPipe
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Отображение видеопотока
        cv2.imshow('Mediapipe Feed', image)

        # Условие для выхода из программы (клавиша 'q')
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Завершение работы с камерой и закрытие всех окон
cap.release()
cv2.destroyAllWindows()
