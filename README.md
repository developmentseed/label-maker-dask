# label-maker-dask

Library for running [label-maker](https://github.com/developmentseed/label-maker/) as a [dask](https://dask.org/) job

# Basic Example

Instantiate a distributed dask cluster
```python
from dask.distributed import Client
cluster = ...
client = Client(cluster)
```

Create a label maker job
```python
from label_maker_dask import LabelMakerJob
lmj = LabelMakerJob(
    zoom=13,
    bounds=[-44.4836425781, -23.02665962797, -43.412719726, -22.5856399016],
    classes=[
        { "name": "Roads", "filter": ["has", "highway"] },
        { "name": "Buildings", "filter": ["has", "building"] }
      ],
    imagery="http://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=ACCESS_TOKEN",
    ml_type="segmentation",
    label_source="https://qa-tiles-server-dev.ds.io/services/z17/tiles/{z}/{x}/{y}.pbf"
)
```

Build & execute the job
```python
lmj.build_job()
lmj.execute_job()
```

View or otherwise use the results (by passing to a machine learning framework)
```python
for result in lmj.results:
    ...
```