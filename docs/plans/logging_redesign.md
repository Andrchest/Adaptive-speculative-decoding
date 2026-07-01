# План: Переработка вывода в консоль

> **Дата:** 2026-06-23  
> **Цель:** Удобный вывод — 3 режима, детальные результаты после каждого эксперимента, прогресс-бар вместо спама логов

---

## Текущая проблема

| Место | Что выводится | Проблемы |
|-------|---------------|----------|
| **Старт** (runner.py) | `Running: name` | ✅ Хвалю |
| **Подготовка** (base.py) | `Loading models`, `GPU mem`, `Rule1/Rule2`, `Cache` | 20+ строк INFO — нет смысла видеть при каждом эксперименте |
| **Во время** (base.py) | `Progress [i/n] acc= TPS=` | Один раз каждые 50 prompt-ов — ок |
| **Конец эксперимента** (collector.py) | `Benchmark summary: n_seq= acc= tps= cache_hit=` | Только 4 метрики из 15+ |
| **Финальная таблица** (main.py) | 3 колонки: Acc Rate, TPS, Speedup | Ничего больше |

**Итог:** ничего не выводится в конец эксперимента (кроме 4 метрик), финальная таблица бедная, во время — `logger.info` съедает CPU/IO.

---

## Решение: 3 режима вывода + расширенный финальный вывод

### Режимы вывода

Определяются флагом `--log-level` (или переменным окружения `LOG_LEVEL`):

| Флаг | Значение по умолчанию | Поведение |
|------|----------------------|-----------|
| `--log-level QUIET` | ✅ | Только прогресс-бар во время, summary в конце |
| `--log-level NORMAL` | — | Прогресс-бар + минимальные логги (warn/error) |
| `--log-level VERBOSE` | — | Текущее поведение — все логги |

**Реализация:** один вызов `logging.basicConfig(level=...)` в `main()`, который устанавливает глобальный filter.

---

## A) Что выводится в конце каждого эксперимента

### Текущий вывод (collector.summary() → logger.info):
```
Benchmark summary for 01_baseline: n_seq=2 acc=0.520 tps=38.9 cache_hit=0.000
```

### Новый вывод (после `collector.summary()`):

```
============================================================
  Experiment: 01_baseline — Rule1 + Rule2 + NgramCache(LRU) + no distillation
============================================================
  Duration: 1.387s  (0.693s per sample, 2 sequences)
  Throughput: 38.9 tok/s  (avg 39.8 tok/s)
  Acceptance: 57.1%  (2.84/5.00 avg accepted / draft)
  Cache hit:   0.0%
  GPU: peak=0.68 GB  mean=0.67 GB
  ───────────────────────────────────────────────────
  ✓ Saved to: results/01_baseline.json
  ✓ CSV updated: results/comparison_table.csv
============================================================
```

**Что добавлено:**
- Название + описание эксперимента
- Wall time (total + mean)
- TPS + avg TPS
- Acceptance rate + avg accepted tokens + avg draft length
- Cache hit rate
- GPU peak/mean
- Пути к сохранённым файлам
- Блок с `✓` — визуально завершение

---

### Для экспериментов с distillation (04-07, 11):
```
  Loss: mean=3.26  kl=0.76  nll=5.60
  Distillation steps: 1
```

### Для экспериментов с router (09, 11):
```
  Router: {mlp: 1.0}
```

---

## B) Что выводится во время эксперимента (прогресс)

### QUIET / NORMAL режим:
```
[████████░░░░░░░░░░░░░░] 2/10  acc=0.520  tps=38.9  0.69s/sample
```
— одна строка в tqdm, обновляется in-place  
— без `logger.info` в цикле, только tqdm bar

### VERBOSE режим:
```
INFO: Progress [1/10] acc=0.520 tps=38.9
INFO: Progress [2/10] acc=0.530 tps=37.5
...
```
— текущее поведение, каждую итерацию (или каждые N итераций)

**Реализация:** tqdm из `tqdm` библиотеки, обёртывающий цикл `for i, (input_ids, prompt_len) in enumerate(prompts)`.

---

## C) Финальная таблица (в конце всех экспериментов)

### Текущая:
```
=== Final Comparison ===
Experiment              Acc Rate          TPS     Speedup
-----------------------------------------------------------------
01_baseline              0.571       38.9        0.00x
```

### Новая:
```
================================================================
  Final Comparison
================================================================
#   Experiment              Acc   TPS    Wall(s)   Acc/Avg   GPU(GB)
---------------------------------------------------------------------------
 1  10_+universal          92.9%  69.4    0.865    4.62/5.0   1.15  ⚡ Fastest
 2  09_+routing            57.1%  43.9    1.231    2.84/5.0   0.68
 3  01_baseline            57.1%  38.9    1.387    2.84/5.0   0.68
 4  02_+lattice            57.1%  36.0    1.500    2.84/5.0   0.69
 5  03_translator          53.0%  32.6    1.532    2.63/5.0   0.80
 6  05_+replay_fifo        47.5%  27.9    1.610    2.25/5.0   2.09
 7  06_+replay_prio        47.5%  27.8    1.621    2.25/5.0   2.09
 8  07_+contrastive        47.5%  27.1    1.661    2.25/5.0   2.09
 9  04_+online_distil      47.5%  26.6    1.689    2.25/5.0   2.09
10  08_+speedup_adapt      25.9%  19.2    2.079    1.43/5.9   0.68
11  11_full_system         27.9%  14.7    2.374    1.21/4.3   2.66  🐌 Slowest
================================================================
  Results: results/01_baseline.json, results/02_+lattice.json, ...
  CSV:     results/comparison_table.csv
================================================================
```

**Что добавлено:**
- Wall time для каждого эксперимента
- GPU peak
- Acceptance / avg accepted (отношение к draft_length)
- Сортировка по wall time (быстрее сверху)
- Бейджи `⚡` (fastest) / `🐌` (slowest)
- Список файлов с результатами

---

## D) Файлы для изменения

| Файл | Изменения | Размер |
|------|-----------|--------|
| `src/main.py` | Добавить `--log-level` flag, `logging.Filter`, обновить `_print_summary` | ~30 строк |
| `src/experiments/base.py` | Заменить `logger.info("Progress...")` на tqdm update, добавить `log_level` в контекст | ~20 строк |
| `src/experiments/runner.py` | Передать `log_level` в runner → experiment | ~5 строк |
| `src/benchmarks/metrics/collector.py` | Добавить `print_end_summary()` метод, который принимает metrics dict и выводит блок | ~50 строк |

**Итого:** ~105 строк изменений в 4 файлах

---

## E) Последовательность реализации

### Шаг 1: Добавить `--log-level` в main.py
- Парсинг аргумента: `QUIET`, `NORMAL`, `VERBOSE`
- Настройка глобального log level: `logging.getLogger().setLevel(...)`
- Добавить фильтр для подавления INFO в QUIET режиме

### Шаг 2: Добавить tqdm прогресс-бар в base.py
- `from tqdm import tqdm`
- Обёртка цикла `for i, (input_ids, prompt_len) in enumerate(tqdm(prompts, desc=exp_name))`
- Удалить `logger.info("Progress [...]")`
- tqdm сам обновляет прогресс на месте (no console spam)

### Шаг 3: Расширенный вывод в конце эксперимента
- В `base.py`, после `summary = collector.summary()`, вызвать `collector.print_end_summary(summary)`
- `collector.print_end_summary()` выводит отформатированный блок с all metrics

### Шаг 4: Обновлённая финальная таблица в main.py
- `_print_summary()` принимает список metrics dicts
- Выводит таблицу с 6+ колонками
- Сортировка по wall time
- Бейджи fastest/slowest

### Шаг 5: Тесты
- Unit-тесты для `print_end_summary()`
- Интеграционный тест: запустить `--log-level QUIET` и проверить отсутствие спам-лога

---

## F) Зависимости

```python
# requirements.txt или pyproject.toml
tqdm  # ← нужно добавить
```

`tqdm` — лёгкая зависимость, ~60KB. Если нельзя добавлять — заменить на простой ASCII прогресс-бар.

---

## G) Риски и mitigation

| Риск | Mitigation |
|------|-----------|
| tqdm не установлен на старом окружении | Optional import: `try: from tqdm import ...` fallback на plain print |
| `logging.basicConfig` конфликтует с существующим | Использовать `logging.getLogger("src").setLevel(...)` вместо глобального |
| tqdm в Jupyter/IDE выводит HTML | tqdm auto-detects env, no action needed |
| Большой финальный вывод на маленьких экранах | Wrap-текст в `rich` (уже есть в проекте) |

---

## H) Что НЕ меняем (out of scope)

- JSON/CSV сохранение — оставляем как есть (полные данные)
- `logger.debug()` — всё ещё идёт в /dev/null при QUIET
- MLflow logging — не зависит от log level
- Test infrastructure — логи тестов unaffected

---

## Итог

| Режим | Старт | Во время | Конец | Финал |
|-------|-------|----------|-------|-------|
| QUIET | `✓ Running: 01_baseline` | tqdm bar | Summary block | Full table |
| NORMAL | `✓ Running: 01_baseline` | tqdm bar | Summary block + warn/error | Full table |
| VERBOSE | Всё как сейчас | Всё как сейчас | Всё как сейчас + summary block | Full table |

Пользователь выбирает один раз через `--log-level`, остальное работает автоматически.
