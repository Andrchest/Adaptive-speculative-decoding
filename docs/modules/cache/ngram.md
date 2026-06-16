# NgramCache

N-gram based cache with pluggable eviction strategies.

## Eviction Strategies

- **LRU** — Least Recently Used
- **LFU** — Least Frequently Used
- **ACC** — acceptance-weighted (hit_count × acceptance_rate / age)
- **Hybrid** — same as ACC

## Interface

```python
cache = NgramCache(max_size=65536, n=3, eviction="hybrid")
entry = cache.lookup(context)        # optional[CacheEntry]
cache.insert(context, token_ids)
cache.update_acceptance(context, accepted)
```
