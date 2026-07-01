# Fix: CUDA OOM in Rule1Mapping.map_logits() — 01_baseline

## Проблема

Эксперимент `01_baseline` с флагом `--tiny` (facebook/opt-125m + facebook/opt-350m)
падает при первом вызове `SpeculativeDecoder.generate()` с ошибкой:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 47.07 GiB.
GPU 0 has a total capacity of 7.66 GiB of which 6.77 GiB is free.
```

Ошибка возникает в `src/core/translation/rules.py:166` в `Rule1Mapping.map_logits()`:
```python
target_probs.index_add_(1, target_d_indices, drafter_probs[:, source_d_indices])
```

## Анализ

Ожидаемый объём используемой памяти: ~3.5 MB (все тендеры в сумме).
Запрашиваемый объём: 47.07 GiB — физически невозможно для данной операции.

Вероятная причина: PyTorch 2.5.1 CUDA allocator bug с `index_add_` в комбинации
с memory fragmentation после загрузки моделей + размер vocab 50272 (не степень двойки).

## План решения

### Фаза 1: Diagnostic — подтвердить корень проблемы

| № | Действие | Ожидаемый результат |
|---|----------|---------------------|
| 1.1 | Добавить print-дебаг перед `index_add_`: shape, dtype, device, nnz | Подтвердить размеры и устройства тендеров |
| 1.2 | Попробовать `scatter_add_` вместо `index_add_` | Если работает — подтверждение бага ядра CUDA |
| 1.3 | Запустить с `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Если работает — проблема в fragmentation аллокатора |
| 1.4 | Попробовать `torch.cuda.empty_cache()` перед decode-loop | Исключить residual fragmentation из model loading |

### Фаза 2: Fix — применить рабочее решение

**Вариант C (рекомендуемый): Прямое присваивание вместо `index_add_`**

В Rule1 каждый drafter token maps к уникальному target token (1:1 mapping),
поэтому аккумуляция не нужна — можно использовать прямое присваивание:

```python
src = drafter_probs[:, source_d_indices]  # (B, M)
indices_expanded = target_d_indices.unsqueeze(0).expand(batch, -1)  # (B, M)
target_probs.scatter_add_(1, indices_expanded, src)
```

Или если индексы уникальны в строке (что всегда верно для Rule1):
```python
target_probs[:, target_d_indices] = src
```

### Фаза 3: Harden — предотвратить повторение

| № | Действие | Цель |
|---|----------|------|
| 3.1 | Добавить assert-shape checks в `map_logits()` | Catch shape mismatches early |
| 3.2 | Использовать `torch.float16` где возможно | Уменьшить peak memory вдвое |
| 3.3 | Настроить `PYTORCH_CUDA_ALLOC_CONF` | Предотвратить fragmentation |
| 3.4 | Добавить OOM fallback: warn → CPU ops → continue | Graceful degradation |

### Фаза 4: Verification — проверить всё работает

| № | Действие | Ожидаемый результат |
|---|----------|---------------------|
| 4.1 | Запустить `--experiment 01_baseline --tiny -n 5` | Без OOM, 5 шагов за 10-30 сек |
| 4.2 | Запустить `--experiment 01_baseline --tiny` (500 samples) | Все 500 сэмплов обработаны |
| 4.3 | Проверить `results/01_baseline.json` | Metrics: Acc Rate > 0, Speedup > 0 |
| 4.4 | Запустить `--suite ablation` (все 11) | Ни один эксперимент не падает по памяти |
| 4.5 | Сравнить metrics до/после fix | No regression в accuracy/speedup |

## Файлы для изменения

1. `src/core/translation/rules.py` — `Rule1Mapping.map_logits()`
2. `src/experiments/runner.py` — опционально: `torch.cuda.empty_cache()` перед loop
3. `.env` / docker config — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Риск

- **Low**: замена одного ядра CUDA на эквивалентную операцию
- Валидация: после fix все 11 ablation experiments должны работать без регрессии метрик
