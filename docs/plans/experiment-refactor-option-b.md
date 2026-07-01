# Plan: Experiment Refactoring — Option B (Strategy Pattern)

**Status:** Draft  
**Author:** AI assistant  
**Date:** 2026-06-21  
**Confidence:** Medium — requirements understood, but approach introduces new abstractions that need validation through proof slice  
**Scope:** Refactor `src/experiments/` from flag-based god-method to Strategy pattern with inheritance  

---

## 1. Problem

`src/experiments/runner.py` (~800 строк) содержит:
- `ExperimentConfig` — dataclass с 30+ флагами (`use_lattice`, `use_translator`, `use_online_distil`, ...)
- `ExperimentRunner._run_one()` — ~300-строчный метод, который по флагам условно строит пайплайн через monkey-patching
- `ABLATION_SUITE` — хардкодный список из 11 конфигов

**Последствия для параллельной работы:**
- Два исследователя в разных ветках правят один и тот же `_run_one()` → merge конфликты
- Новый компонент = правка 3-4 мест (Config + Runner + __init__ + main.py)
- Невозможно добавить эксперимент с принципиально иной логикой (не просто «ещё один флаг»)
- Тестирование отдельных компонентов в контексте эксперимента затруднено

---

## 2. Goal

Каждый эксперимент — самостоятельный класс, наследующий `BaseExperiment`. Исследователь в своей ветке создаёт файл с классом и регистрирует его — без правки общего кода.

```
# researcher/ivan/my_experiment.py  (новая ветка, нет конфликтов)
class IvansNovelExperiment(BaseExperiment):
    def build_pipeline(self, ctx: BuildContext) -> Pipeline:
        return Pipeline(stages=[...])
```

**Anti-goals:**
- Это НЕ миграция на Pipeline-архитектуру (Option C) — шаги decode остаются в `SpeculativeDecoder`
- Это НЕ переписывание core компонентов — `DraftModel`, `TargetModel`, `CrossVocabTranslator` и т.д. остаются как есть
- Мы не убираем `ExperimentConfig` полностью — он становится базовым конфигом, который подклассы могут расширять

---

## 3. Architecture

### 3.1 Target Structure

```
src/experiments/
├── __init__.py                  # public API: BaseExperiment, ExperimentRunner, suites
├── base.py                      # BaseExperiment (ABC), ExperimentResult, BuildContext
├── pipeline.py                  # Pipeline, Component (stages inside one experiment)
├── runner.py                    # ExperimentRunner (orchestrator only, ~150 строк)
├── suites.py                    # ABLATION_SUITE, CACHE_SUITE, DATASET_SUITE
├── built_in/                    # встроенные эксперименты из текущего _run_one()
│   ├── __init__.py
│   ├── baseline.py
│   ├── with_lattice.py
│   ├── with_translator.py
│   ├── with_online_distil.py
│   ├── with_replay.py
│   ├── with_contrastive.py
│   ├── with_speedup_adapt.py
│   ├── with_routing.py
│   ├── with_universal.py
│   └── full_system.py
└── templates/                   # заготовки для исследователей
    ├── __init__.py
    └── minimal_template.py      # copy-paste шаблон
```

### 3.2 Class Hierarchy

```
BaseExperiment (ABC)
├── BaselineExperiment
├── LatticeExperiment          (extends BaselineExperiment)
├── TranslatorExperiment       (extends LatticeExperiment)
├── OnlineDistillExperiment
├── ReplayExperiment           (extends OnlineDistillExperiment)
├── ContrastiveExperiment      (extends OnlineDistillExperiment)
├── SpeedupAdaptiveExperiment
├── RoutingExperiment
├── UniversalDrafterExperiment
└── FullSystemExperiment

Исследовательский эксперимент:
├── IvansNovelExperiment       (extends BaseExperiment или любой built-in)
```

### 3.3 Key Interfaces

#### `BaseExperiment`

```python
@dataclass
class ExperimentMeta:
    """Metadata visible to runner and MLflow."""
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    # Which ablation dimensions this experiment touches
    dimensions: list[str] = field(default_factory=list)
    # Dependencies on other experiments (for ordering)
    depends_on: list[str] = field(default_factory=list)


class BaseExperiment(ABC):
    """Base class for all experiments.

    Subclasses override methods to customize:
    - which models to load
    - which components to build
    - how to run the decode loop
    - how to collect additional metrics
    """

    def __init__(self, meta: ExperimentMeta | None = None):
        self.meta = meta or ExperimentMeta(name=self.__class__.__name__)
        self._components: dict[str, Any] = {}

    # --- Lifecycle (override these) ---

    @abstractmethod
    def get_config(self) -> ExperimentConfig:
        """Return the base config (models, dataset, hyperparams)."""
        ...

    def build_translator(self, ctx: BuildContext) -> CrossVocabTranslator:
        """Build and return the translator. Default: Rule1+Rule2."""
        ...

    def build_cache(self, ctx: BuildContext) -> NgramCache:
        """Build and return the cache. Default: NgramCache(hybrid)."""
        ...

    def build_distiller(self, ctx: BuildContext) -> OnlineDistiller | None:
        """Build and return distiller. Default: None."""
        ...

    def build_adaptive_controller(self, ctx: BuildContext):
        """Build adaptive draft controller. Default: None."""
        ...

    def build_router(self, ctx: BuildContext):
        """Build dynamic router. Default: None."""
        ...

    def build_universal_drafter(self, ctx: BuildContext):
        """Build universal drafter adapter. Default: None."""
        ...

    def on_before_decode(self, ctx: DecodeContext) -> None:
        """Hook before decode loop starts."""
        pass

    def on_decode_step(self, ctx: DecodeContext, step_result: StepResult) -> None:
        """Hook after each decode step. For distillation, routing, etc."""
        pass

    def on_after_decode(self, ctx: DecodeContext) -> None:
        """Hook after decode loop finishes."""
        pass

    def on_extra_metrics(self, summary: dict) -> dict:
        """Add experiment-specific metrics to the summary. Default: pass-through."""
        return summary

    # --- Execution (usually don't override) ---

    def run(self, runner: ExperimentRunner) -> ExperimentResult:
        """Full experiment lifecycle. Orchestrates build → run → collect."""
        ...
```

#### `BuildContext` / `DecodeContext`

```python
@dataclass
class BuildContext:
    """Shared context during component construction."""
    device: str
    drafter: DraftModel
    target: TargetModel
    config: ExperimentConfig
    # Components built so far (for dependencies)
    components: dict[str, Any]


@dataclass
class DecodeContext:
    """Shared context during the decode loop."""
    decoder: SpeculativeDecoder
    collector: BenchmarkCollector
    config: ExperimentConfig
    # Mutable state that hooks can read/write
    distiller: OnlineDistiller | None = None
    router: DynamicRouter | None = None
    adaptive_fn = None
    extra_state: dict[str, Any] = field(default_factory=dict)
```

#### `ExperimentResult`

```python
@dataclass
class ExperimentResult:
    meta: ExperimentMeta
    config: dict          # asdict(config)
    metrics: dict         # collector.summary() + extra
    error: str | None = None
```

### 3.4 Runner (simplified)

```python
class ExperimentRunner:
    """Orchestrates a list of BaseExperiment instances."""

    def __init__(self, experiments: list[BaseExperiment], ...):
        self.experiments = experiments

    def run_all(self) -> list[ExperimentResult]:
        for exp in self.experiments:
            self._clear_gpu()
            result = exp.run(self)    # each experiment knows how to run itself
            self._save_result(result)
        self._write_csv(results)

    def _build_models(self, cfg: ExperimentConfig) -> tuple[DraftModel, TargetModel]:
        """Shared model loading logic."""
        ...

    def _load_dataset(self, ...) -> list:
        """Shared dataset loading (from current _load_dataset)."""
        ...
```

### 3.5 Registration (Discovery)

```python
# experiments/suites.py
def discover_experiments() -> list[BaseExperiment]:
    """Auto-discover experiments from built_in/ and research/ directories."""
    experiments = []

    # Built-in experiments
    from experiments.built_in import __all__ as built_in_names
    for name in built_in_names:
        experiments.append(_import_experiment(f"experiments.built_in.{name}"))

    # Research experiments (from research/*/experiments/*.py)
    research_dir = Path(__file__).resolve().parents[2] / "research"
    for exp_file in research_dir.rglob("experiments/*.py"):
        if exp_file.name.startswith("_"):
            continue
        experiments.append(_import_experiment_from_path(exp_file))

    return experiments


ABLATION_SUITE = [
    BaselineExperiment(),
    LatticeExperiment(),
    TranslatorExperiment(),
    # ... etc
]
```

---

## 4. Task List

### Phase 0: Proof Slice — интерфейсы без миграции

**Цель:** Убедиться, что абстракции работают, на одном эксперименте (baseline).

- [ ] **0.1** Создать `src/experiments/base.py` с ABC `BaseExperiment`, `ExperimentMeta`, `BuildContext`, `DecodeContext`, `ExperimentResult`
  - **Acceptance:** mypy strict pass, все публичные методы имеют docstrings + type hints

- [ ] **0.2** Создать `src/experiments/built_in/baseline.py` — `BaselineExperiment`, реализующий текущий baseline (Rule1+Rule2, no distillation)
  - **Acceptance:** `BaselineExperiment().run(runner)` возвращает те же метрики, что текущий `01_baseline` из ABLATION_SUITE

- [ ] **0.3** Создать упрощённый `ExperimentRunner.run_one_experiment()` который вызывает `exp.run(runner)` вместо `_run_one(cfg)`
  - **Acceptance:** `python src/main.py --smoke` работает с новым runner-ом (параллельно со старым)

- [ ] **0.4** Написать unit-тест: `BaselineExperiment` даёт идентичные метрики со старым конфигом
  - **Acceptance:** `pytest tests/unit/test_baseline_equivalence.py` passes

**Exit criteria:** Один эксперимент (baseline) работает через новую архитектуру и даёт те же результаты.

---

### Phase 1: Миграция встроенных экспериментов

**Цель:** Перенести все 11 экспериментов из ABLATION_SUITE в отдельные классы.

- [ ] **1.1** `LatticeExperiment` — replace Rule2 with TokenizerLattice
  - Override: `build_translator()`
  - **Acceptance:** Метрики == `02_+lattice`

- [ ] **1.2** `TranslatorExperiment` — add learned TranslatorModel
  - Override: `build_translator()` (extends LatticeExperiment)
  - **Acceptance:** Метрики == `03_+translator`

- [ ] **1.3** `OnlineDistillExperiment` — add OnlineDistiller
  - Override: `build_distiller()`, `on_decode_step()`
  - **Acceptance:** Метрики == `04_+online_distil`

- [ ] **1.4** `ReplayExperiment` — add ReplayBuffer (fifo + prioritized variants)
  - Override: `build_distiller()` (wraps with ReplayBuffer), `on_decode_step()`
  - Parameterize: `strategy: Literal["fifo", "prioritized"]`
  - **Acceptance:** Метрики == `05_+replay_fifo` и `06_+replay_prio`

- [ ] **1.5** `ContrastiveExperiment` — add ContrastiveLoss
  - Override: `build_distiller()` (with contrastive), `on_decode_step()`
  - **Acceptance:** Метрики == `07_+contrastive`

- [ ] **1.6** `SpeedupAdaptiveExperiment` — add SpeedupPredictor
  - Override: `build_adaptive_controller()`
  - **Acceptance:** Метрики == `08_+speedup_adapt`

- [ ] **1.7** `RoutingExperiment` — add DynamicRouter
  - Override: `build_router()`, `on_decode_step()` (select drafter per prompt)
  - **Acceptance:** Метрики == `09_+routing`

- [ ] **1.8** `UniversalDrafterExperiment` — add UniversalDrafter
  - Override: `build_universal_drafter()` (adapter pattern для drafter)
  - **Acceptance:** Метрики == `10_+universal`

- [ ] **1.9** `FullSystemExperiment` — все компоненты вместе
  - Наследуется от базового класса и override-ит всё
  - **Acceptance:** Метрики == `11_full_system`

- [ ] **1.10** Обновить `experiments/suites.py` — ABLATION_SUITE создаёт экземпляры классов
  - **Acceptance:** `python src/main.py --suite ablation --tiny -n 1` прогоняет все 11

**Exit criteria:** Все 11 экспериментов migrated, метрики идентичны старым (в пределах floating-point noise).

---

### Phase 2: Удаление старого кода и обновление entry points

**Цель:** Убрать `_run_one()`, `ExperimentConfig` как единственный способ запуска, обновить main.py.

- [ ] **2.1** Обновить `ExperimentRunner`:
  - Убрать `_run_one(cfg: ExperimentConfig)` — заменить на `run_one_experiment(exp: BaseExperiment)`
  - Убрать `_apply_lora()` — перенести в `OnlineDistillExperiment.build_distiller()`
  - Сохранить `_clear_gpu_memory()`, `_load_dataset()`, `_save_result()`, `_write_csv()` как утилиты раннера
  - **Acceptance:** Runner ~150 строк (было ~800)

- [ ] **2.2** Обновить `src/main.py`:
  - `--suite ablation` → использует `suites.ABLATION_SUITE` (список экземпляров)
  - `--experiment name` → поиск по `meta.name` среди всех зарегистрированных
  - `--list` → показывает все discovered experiments с description и tags
  - `--smoke` → `SmokeTestExperiment()` (подкласс BaseExperiment)
  - **Acceptance:** Все CLI команды работают

- [ ] **2.3** Удалить `ABLATION_SUITE` из `runner.py` (перенесено в `suites.py`)
  - **Acceptance:** `runner.py` не содержит хардкодных конфигов

- [ ] **2.4** Обновить `experiments/__init__.py` — экспортировать публичный API:
  ```python
  from .base import BaseExperiment, ExperimentMeta, BuildContext, DecodeContext, ExperimentResult
  from .runner import ExperimentRunner
  from .suites import ABLATION_SUITE, discover_experiments
  ```
  - **Acceptance:** `from experiments import BaseExperiment` работает

**Exit criteria:** Старый `_run_one()` удалён, CLI работает через новую архитектуру.

---

### Phase 3: Инфраструктура для исследователей

**Цель:** Сделать тривиальным добавление новых экспериментов в research ветках.

- [ ] **3.1** Создать `experiments/templates/minimal_template.py`:
  ```python
  """Copy this file to research/<your-name>/experiments/ and customize."""
  from experiments import BaseExperiment, BuildContext, DecodeContext, ExperimentConfig, ExperimentMeta

  class MyExperiment(BaseExperiment):
      def __init__(self):
          super().__init__(ExperimentMeta(
              name="my_experiment",
              description="One-line description",
              tags=["translation", "cache"],
              dimensions=["translation_strategy"],
          ))

      def get_config(self) -> ExperimentConfig:
          cfg = ExperimentConfig(name=self.meta.name)
          # Override params here
          return cfg

      def build_translator(self, ctx: BuildContext):
          # Custom translator logic
          translator = super().build_translator(ctx)
          # Modify translator
          return translator
  ```
  - **Acceptance:** Файл содержит комментарии-подсказки на каждом override-методе

- [ ] **3.2** Реализовать `discover_experiments()` — авто-поиск экспериментов в `research/*/experiments/`
  - Динамический import через `importlib.util.module_from_spec()`
  - Graceful handling ошибок (битый эксперимент не ломает весь список)
  - **Acceptance:** Эксперимент из template обнаруживается и появляется в `--list`

- [ ] **3.3** Добавить `--research` флаг в main.py:
  - `python src/main.py --research ivan` → запускает эксперименты из `research/ivan/experiments/`
  - `python src/main.py --research all` → все research эксперименты
  - **Acceptance:** Research эксперименты запускаются через CLI

- [ ] **3.4** Создать `research/README.md` с гайдлайнами:
  - Как создать новый эксперимент (copy template → customize → run)
  - Как наследовать от built-in экспериментов
  - Как добавить кастомные метрики через `on_extra_metrics()`
  - Как использовать hooks (`on_before_decode`, `on_decode_step`, `on_after_decode`)
  - **Acceptance:** Гайд читаем для нового члена команды

**Exit criteria:** Исследователь может добавить новый эксперимент за 5 минут без правки общего кода.

---

### Phase 4: Тесты и валидация

**Цель:** Гарантировать, что рефакторинг не сломал поведение.

- [ ] **4.1** Differential tests: каждый migrated experiment даёт те же метрики, что старый конфиг
  - Фикстуры: сохранить JSON результатов старого runner-а для 1-sample run
  - Assert: все метрики в пределах 1% (floating-point tolerance)
  - **Acceptance:** `pytest tests/unit/test_experiment_equivalence.py -v` — 11 tests pass

- [ ] **4.2** Unit-тесты для base classes:
  - `BaseExperiment` без override падает с `NotImplementedError` на `get_config()`
  - `BuildContext` корректно передаёт компоненты между build-методами
  - **Acceptance:** `pytest tests/unit/test_experiment_base.py` passes

- [ ] **4.3** Integration test: полный прогон `--suite ablation --tiny -n 1`
  - **Acceptance:** Все 11 экспериментов завершаются без ошибок

- [ ] **4.4** Regression: `--smoke` тест
  - **Acceptance:** Smoke test проходит < 2 минуты

**Exit criteria:** Все тесты green, метрики эквивалентны старым.

---

### Phase 5: Очистка и документация

**Цель:** Код чистый, документация обновлена.

- [ ] **5.1** `ExperimentConfig` — оставить как dataclass для базовых параметров, но добавить docstring что он используется внутри `BaseExperiment.get_config()`
- [ ] **5.2** Обновить `AGENTS.md` — добавить раздел об архитектуре экспериментов
- [ ] **5.3** Обновить docstring в `experiments/__init__.py` с примером использования
- [ ] **5.4** `ruff check` + `ruff format` + `mypy` — всё чистое
- [ ] **5.5** Удалить `__pycache__` и артефакты

---

## 5. Dependency Graph

```
Phase 0 (Proof Slice)
    ↓
Phase 1 (Migrate 11 experiments)    ← tasks 1.1-1.8 independent, 1.9 depends on all
    ↓
Phase 2 (Remove old code)           ← depends on Phase 1 complete
    ↓
Phase 3 (Researcher infrastructure)  ← depends on Phase 2 (needs clean API)
    ↓
Phase 4 (Tests)                      ← can start after Phase 1, finish after Phase 3
    ↓
Phase 5 (Cleanup)                    ← depends on all
```

**Parallel opportunities:**
- Tasks 1.1–1.8 можно делать параллельно (разные файлы)
- Phase 3.1 (template) можно делать параллельно с Phase 1
- Phase 4.1-4.2 можно начинать после Phase 0

---

## 6. Scenarios: Работа с новой структурой

### Сценарий 1: Исследователь добавляет новый компонент

**Иван** хочет попробовать новый метод транслации `PhoneticMapping`.

```bash
# 1. Создаёт директорию (если ещё нет)
mkdir -p research/ivan/experiments

# 2. Копирует шаблон
cp src/experiments/templates/minimal_template.py \
   research/ivan/experiments/phonetic_translation.py

# 3. Редактирует:
```

```python
# research/ivan/experiments/phonetic_translation.py
from experiments import BaseExperiment, BuildContext, ExperimentConfig, ExperimentMeta
from experiments.built_in.baseline import BaselineExperiment
from my_research.phonetic import PhoneticMapping  # свой модуль

class PhoneticTranslationExperiment(BaselineExperiment):
    """Baseline + PhoneticMapping instead of Rule2."""

    def __init__(self):
        super().__init__(ExperimentMeta(
            name="phonetic_translation",
            description="Replace Rule2 with phonetic character-level mapping",
            tags=["translation", "ivan"],
            dimensions=["translation_strategy"],
        ))

    def build_translator(self, ctx: BuildContext):
        # Берём базовый translator (Rule1 + Rule2)
        translator = super().build_translator(ctx)
        # Заменяем Rule2 на PhoneticMapping
        translator.rule2 = PhoneticMapping(
            ctx.drafter.tokenizer,
            ctx.target.tokenizer,
        )
        return translator
```

```bash
# 4. Запускает
python src/main.py --research ivan -n 50 --tiny

# 5. Смотрит результаты
cat results/phonetic_translation.json
```

**Merge:** Когда Иван делает PR, в `src/experiments/` нет конфликтов — только новый файл в `research/ivan/`.

---

### Сценарий 2: Исследователь хочет изменить decode loop

**Мария** хочет попробовать другой порядок: сначала cache lookup, потом draft (вместо draft → cache).

```python
# research/maria/experiments/cache_first.py
from experiments import BaseExperiment, DecodeContext, ExperimentMeta
from experiments.built_in.baseline import BaselineExperiment

class CacheFirstExperiment(BaselineExperiment):
    """Check cache BEFORE drafting — avoids unnecessary draft when cache hits."""

    def __init__(self):
        super().__init__(ExperimentMeta(
            name="cache_first",
            description="Cache lookup before draft generation",
            tags=["cache", "maria"],
        ))

    def on_decode_step(self, ctx: DecodeContext, step_result):
        # Кастомная логика: если cache hit, skip distillation
        if step_result.cache_hit and ctx.distiller:
            ctx.extra_state["skipped_distill"] = \
                ctx.extra_state.get("skipped_distill", 0) + 1

    def on_extra_metrics(self, summary: dict) -> dict:
        # (нужно хранить счётчик где-то доступный)
        summary["skipped_distill_steps"] = self._skipped_count
        return summary
```

---

### Сценарий 3: Сравнение нескольких вариантов одного компонента

**Егор** хочет сравнить 3 стратегии eviction для cache.

```python
# research/egor/experiments/cache_strategies.py
from experiments import BaseExperiment, ExperimentConfig, ExperimentMeta
from experiments.built_in.baseline import BaselineExperiment

class CacheStrategyExperiment(BaselineExperiment):
    """Parameterized cache eviction strategy experiment."""

    def __init__(self, strategy: str):
        super().__init__(ExperimentMeta(
            name=f"cache_{strategy}",
            description=f"N-gram cache with {strategy} eviction",
            tags=["cache", "egor"],
            dimensions=["cache_eviction"],
        ))
        self._strategy = strategy

    def get_config(self) -> ExperimentConfig:
        cfg = super().get_config()
        cfg.cache_eviction = self._strategy
        return cfg

# suites.py или main.py регистрирует:
CACHE_STRATEGY_SUITE = [
    CacheStrategyExperiment("lru"),
    CacheStrategyExperiment("lfu"),
    CacheStrategyExperiment("acc"),
    CacheStrategyExperiment("hybrid"),
]
```

```bash
python src/main.py --research egor -n 100
```

---

### Сценарий 4: Два исследователя работают параллельно

```
main branch:
  src/experiments/
    base.py, runner.py, suites.py, built_in/

feature/ivan-phonetic  (от main):
  research/ivan/experiments/phonetic_translation.py
  research/ivan/phonetic.py  (кастомная логика)
  → NO changes to src/experiments/

feature/maria-cache-first  (от main):
  research/maria/experiments/cache_first.py
  → NO changes to src/experiments/

Мерж:
  main ← feature/ivan-phonetic   ✅ нет конфликтов
  main ← feature/maria-cache-first  ✅ нет конфликтов
```

---

### Сценарий 5: Командный ablation run

Лидер команды хочет прогнать все эксперименты (built-in + research):

```bash
# Все встроенные
python src/main.py --suite ablation

# Все research
python src/main.py --research all

# Конкретный исследователь
python src/main.py --research ivan

# Сравнение в CSV
cat results/comparison_table.csv
```

---

### Сценарий 6: Наследование от другого эксперимента

**Вика** хочет взять `OnlineDistillExperiment` и добавить свой contrastive loss variant:

```python
# research/vika/experiments/improved_contrastive.py
from experiments.built_in.with_contrastive import ContrastiveExperiment

class ImprovedContrastiveExperiment(ContrastiveExperiment):
    """Same as Contrastive but with temperature-scaled KL."""

    def __init__(self):
        super().__init__(ExperimentMeta(
            name="improved_contrastive",
            description="Temperature-scaled contrastive loss",
            tags=["distillation", "contrastive", "vika"],
        ))

    def build_distiller(self, ctx: BuildContext):
        distiller = super().build_distiller(ctx)
        # Override contrastive temperature
        distiller.contrastive_temperature = 2.0  # новый параметр
        return distiller
```

---

## 7. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Метрики отличаются после рефакторинга | Medium | High | Phase 0 proof slice + differential tests (Phase 4.1) |
| Monkey-patch зависимости пропущены | Medium | High | Code review: каждый migrated experiment проверяется на идентичность |
| Исследователи не используют новую архитектуру | Low | Medium | Template + README + пример в research/ — lowering barrier to entry |
| `BuildContext.components` становится god-object | Medium | Low | Явные типы: `components["translator"]` → `ctx.translator` в dataclass |
| Назад не兼容 со старыми скриптами | Low | Low | `ExperimentConfig` остаётся как dataclass; старые скрипты можно обернуть adapter-ом |
| Рефакторинг занимает слишком много времени | Medium | Medium | Phase 0 даёт early validation; если proof slice не работает — остановиться |

---

## 8. Assumptions

| Assumption | Status | Evidence |
|------------|--------|----------|
| Текущие 11 экспериментов покрывают все паттерны использования | Verified | Проанализирован `_run_one()` — все ветки if соответствуют флагам в Config |
| `cache_ablation.py` и `dataset_ablation.py` из main.py ещё не существуют | Verified | `find` не нашёл эти файлы — код в main.py будет падать |
| Исследователи используют Python 3.10+ | Verified | `__pycache__` содержит `.cpython-310.pyc` и `.cpython-312.pyc` |
| GPU memory clearing между экспериментами достаточно текущего `_clear_gpu_memory()` | Verified | Уже работает для 11 последовательных экспериментов |
| Исследователи хотят наследование, а не composition | Unverified | Выбор Option B — если исследователи предпочтут pipeline (Option C), потребуется доп. работа |

---

## 9. Effort Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| 0: Proof Slice | 4 | 1 day |
| 1: Migrate 11 experiments | 10 | 2-3 days (parallelizable) |
| 2: Remove old code | 4 | 0.5 day |
| 3: Researcher infra | 4 | 1 day |
| 4: Tests | 4 | 1 day |
| 5: Cleanup | 5 | 0.5 day |
| **Total** | **31** | **~6 person-days** (2-3 calendar days с 2 параллельными разработчиками) |

---

## 10. Rollout Rule

1. Phase 0 merged → архитектура валидирована на baseline
2. Phase 1 merged → все built-in эксперименты работают через новую систему
3. **Feature flag:** старый `_run_one()` сохраняется до Phase 2 с `DEPRECATED` warning
4. Phase 2 merged → старый код удалён, CLI переключён
5. Phase 3+ → исследователи начинают использовать новую систему

---

## 11. References

- Текущий код: `src/experiments/runner.py` (800 строк)
- Core компоненты: `src/core/models/drafter.py`, `src/core/decoder/speculative.py`
- Extensions: `src/core/extensions/{lattice,translator,replay,routing,multitarget,adaptive,contrastive}/`
- Metrics: `src/benchmarks/metrics/collector.py`
- CLI: `src/main.py`
- Research dirs: `research/{a.polevoi, da.popov, e.pestrovskii, v.poponnikov, m.krylov, al.khadeeva}/`
