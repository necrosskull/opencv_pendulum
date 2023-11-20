import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Загрузка видео
cap = cv2.VideoCapture("ball.mp4")
center = None
# Хранилище координат центра шара
trajectory = []

while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break  # Выход из цикла, если кадры закончились

    # Преобразование кадра в цветовое пространство HSV
    blur = cv2.GaussianBlur(frame, (11, 31), 0)
    # Преобразование кадра в цветовое пространство HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Определение красного цвета в диапазоне HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Нахождение контуров объектов на изображении
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Площадь контура должна быть достаточно большой
        if cv2.contourArea(contour) > 100:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)

                # Отображение контура и центра масс шара
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
            # Отслеживание прямоугольника вокруг контура
            x, y, w, h = cv2.boundingRect(contour)

            # Отображение прямоугольника и центра шара
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = (int(x + w / 2), int(y + h / 2))

            # Добавление координат центра шара в хранилище траектории
            trajectory.append(center)

            # Проверка положения шара и реакция на столкновение с правой стеной
            if center[0] >= 264:
                cv2.putText(
                    frame,
                    f"Collision with wall!",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    # Отображение траектории с использованием векторов и инвертированных цветов
    if len(trajectory) > 10:
        trajectory_array = np.array(trajectory)
        speeds = np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1)
        max_speed = np.max(speeds)
        normalized_speeds = speeds / max_speed

        for i in range(1, len(trajectory_array)):
            color = (
                int(
                    255 - normalized_speeds[i - 1] * 255
                ),  # инверсия цвета при ускорении
                0,
                int(normalized_speeds[i - 1] * 255),  # инверсия цвета при замедлении
            )
            cv2.line(
                frame,
                tuple(trajectory_array[i - 1]),
                tuple(trajectory_array[i]),
                color,
                1,
            )

    # Вывод координат центра шара
    if center:
        cv2.putText(
            frame,
            f"Coordinates: (X:{center[0]} Y:{center[1]})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Отображение кадра
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("HSV", hsv)

    # Замедление видео на 20%
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Создание графика
fig, ax = plt.subplots(figsize=(10, 6))
x_coordinates = [point[0] for point in trajectory]
y_coordinates = [point[1] for point in trajectory]
(line,) = ax.plot(x_coordinates, label="X Coordinate")
(line2,) = ax.plot(y_coordinates, label="Y Coordinate")
ax.set_xlabel("Frame Number")
ax.set_ylabel("Coordinate Value")
ax.set_title("Coordinate Change Over Time")
ax.legend()


# Функция для обновления графика
def update(frames):
    line.set_data(range(frames), x_coordinates[:frames])
    line2.set_data(range(frames), y_coordinates[:frames])
    return line, line2


# Создание анимации
animation = FuncAnimation(
    fig, func=update, frames=len(trajectory), interval=20, repeat=False
)

# Сохранение анимации в файл
animation.save("animation.gif", writer="pillow", fps=30)

# Отображение графика
plt.show()
