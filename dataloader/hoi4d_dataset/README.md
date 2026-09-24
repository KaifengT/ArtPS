

1. Run `dataloader/hoi4d_dataset/hoi4d_gen_objs_matrix.py` and generate `not_ok_ins.txt`
2. Set the path of `not_ok_ins.txt` and category to `dataloader/hoi4d_dataset/hoi4d_prepare_valid_datalists.py`, run it.

3. If you want to use cache, first run
```
hoi4d = HOI4D('path/to/HOI4D', sdf_mode=False, category='C*', mode='train/test', add_noise=True, debug=False, force_write_cache=True, use_cache=True, cache_dir='path/to/your/cache_dir')
for i in tqdm.tqdm(hoi4d):
    ...
```
to generate cache, then you should reconstruct a new HOI4D instance with `force_write_cache=False, use_cache=True, cache_dir='path/to/your/cache_dir'`, this new instance will load from cache.

4. If you don't want to use cache, just run
```
hoi4d = HOI4D('path/to/HOI4D', sdf_mode=False, category='C*', mode='train/test', add_noise=True, debug=False, force_write_cache=False, use_cache=False)

```

