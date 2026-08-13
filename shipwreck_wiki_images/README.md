# Shipwreck Wiki Images

CASTOR dataset — ~110 shipwreck images organized by disaster category.

```
sorted_images/
  aground/     (~42 images)
  capsized/    (~19 images)
  on_fire/     (~16 images)
  sunken/      (~33 images)
```

Images are tracked in git. After cloning BenchyBench with `--recurse-submodules`
the images are available immediately at `shipwreck_wiki_images/sorted_images/`.

All inference scripts resolve images relative to the BenchyBench root, so
on the cluster the expected path is:
`~/BenchyBench/shipwreck_wiki_images/sorted_images/`
