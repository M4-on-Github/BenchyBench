# Shipwreck Wiki Images

CASTOR dataset — ~110 shipwreck images organized by disaster category.

```
sorted_images/
  aground/     (~42 images)
  capsized/    (~19 images)
  on_fire/     (~16 images)
  sunken/      (~33 images)
```

Images are not tracked in git (binary files). Transfer to cluster via rsync:

```bash
rsync -avz shipwreck_wiki_images/sorted_images/ \
    <user>@head1.condo.cs.cmu.edu:~/benchybench/shipwreck_wiki_images/sorted_images/
```

All inference scripts resolve images relative to the benchybench root, so
on the cluster the expected path is:
`~/benchybench/shipwreck_wiki_images/sorted_images/`
