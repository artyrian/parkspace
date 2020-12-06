# Parkspace

Проект по поиск свободных парковочных мест по камерам сверху на основе размеченных заранее доступных мест для парковки.
ML: https://github.com/matterport/Mask_RCNN

### Предобученная модель: 
- скачать в каталог `./models`: [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)

Пример запуска на предобученной модели, baseline:
![baseline](samples/sample_img_maskrcnn_detect_baseline.jpg)

## Сбор датасета

Сохранение изображений с камеры в настоящий момент.
```python src/devline.py --url <server_url> --port <server_port --user <user> --password <password> --dir cams```

Если нет возможности подключиться к серверу видеонаблюдения, можно собрать первый датасет для разметки и проверки вручную.

# Первый запуск

Запуск с предобученной моделью:

```
python detect.py
```

по умолчанию подхватит базовую модель `models/mask_rcnn_coco.h5` для распознвания объектов.
- Вход: `dataset/data/` - изображения по маске *.jpg внутри каталога 
- Результат: `out/`


# Обучение

## Разметка
- собрать train, val выборки
- разметить при помощи via.html
- подложить датасет и разметку по следующим путям

## Обучение на размеченных данных
```
python src/train.py --dataset=dataset --weights=coco
```

## Использование новой модели

tbd


[link]: http://www.ya.ru

[test]: http://ya.ru