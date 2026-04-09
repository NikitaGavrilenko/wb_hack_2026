# Warehouse Shipment Forecasting — 6th Place Solution

Хакатон по прогнозированию объёмов отгрузок со складов. **Итог: 6-е место, score 0.327.**

## Задача

Предсказать `target_1h` (объём отгрузок по маршруту за час) для ~1000 маршрутов на горизонт 4 часа (8 шагов × 30 мин). Метрика: `WAPE + |Relative Bias|`.

## Решение

**Ключевая идея:** 8 отдельных LightGBM моделей (direct multi-step), seed ensemble × 3, temporal ensemble по последним 4 точкам train, per-step bias correction.

### Финальная архитектура (LightGBM v7)

| Параметр | Значение |
|---|---|
| n_estimators | 6000 |
| num_leaves | 127 |
| max_depth | 8 |
| learning_rate | 0.03 |
| seeds | 42, 123, 777 |
| temporal ensemble weights | [0.4, 0.3, 0.2, 0.1] |
| bias correction window | 3 дня |

### Фичи

- **Лаги:** 1, 2, 3, 4, 6, 8, 12, 16, 48, 336 шагов
- **Rolling:** mean/std на 4, 8, 48 шагах
- **Статусы:** current/prev sum, ratio, roll4
- **Route stats:** mean, std, p10, p90
- **Route×hour:** `route_hour_mean`, `vs_route_hour`

### Что пробовали и не сработало

| Эксперимент | LB | Вывод |
|---|---|---|
| MAE objective | 0.650 | Несовместим с bias correction через среднее |
| Recursive multi-step | 0.415 | Накопление ошибок, direct значительно лучше |
| num_leaves=255, max_depth=10 | 0.350 | Переобучение |
| 6 seeds вместо 3 | 0.330 | Плато после 3 сидов |
| PipelineNet (PyTorch) | 0.344 | LightGBM оказался сильнее |
| Blend 90% LGB + 10% NN | 0.329 | Хуже чистого LightGBM |

## Данные

Данные предоставлены организаторами хакатона, в репозиторий не включены.
Положите `train_solo_track.parquet`, `test_solo_track.parquet`, `sample_submission.csv`
в папку `data/` перед запуском ноутбуков.

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook notebooks/03_lgbm_v7_final.ipynb
```

## Структура

```
notebooks/01_eda.ipynb          — EDA, анализ статусов и сезонности
notebooks/02_baseline.ipynb     — Ridge baseline
notebooks/03_lgbm_v7_final.ipynb — финальное решение
notebooks/04_pipelinenet.ipynb  — эксперимент с нейронной сетью
```

## Ключевые выводы

1. Локальная валидация не коррелирует с публичным LB — единственный сигнал это сабмит
2. Direct multi-step значительно лучше recursive из-за накопления ошибок
3. LightGBM с правильными lag-фичами сложно превзойти нейросетями за ограниченное время соревнования