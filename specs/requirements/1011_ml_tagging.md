# Feature Requirement Document: ML Model

I want to create a multi-label, supervised ML song tagger. Please create an implementation plan
under `specs/plans` following the same name as this file.

## Dataset

I have the following dataset:

```python
Schema([('ContentID', String),
        ('FolderPath', String),
        ('Title', String),
        ('ArtistName', String),
        ('BPM', Int32),
        ('Length', Int32),
        ('MyTagNames', List(String)),
        ('MyTagIDs', List(String))])
```

Where all label tags can be found in `data/unique_tags.csv`. I might filter some tags like country, year, as this isn't really something you can extrapolate.

The dataset has the following properties:

- Songs with tags present:  714
- Songs without tags present: 1165
- Statistics: Amount of Tags per Song:

```text
mean: 13.387955
std: 5.309963
median: 13.0
min: 1
max: 38
p25: 10.0
p75: 17.0
p90: 20.0
p95: 21.0
```

The current notebook for a simple tagging approach can be found in `notebooks/03_tag_inference.ipynb`.

## What to explore

Features:

- BPM will probably correlate heavily with some genre based tags.
- Length might be telling, but not really at the same time. 
- Title and artist name will probably contain some information when embedded.
- I think the most juice can be extracted by:
  - loading the mp3 file from the path
  - converting to mel spectrogram or something
  - (or through a deep encoder on hugging face)
  - and using a representation of that to learn a mapping on the superveried set. 
