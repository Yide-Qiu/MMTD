from collections import namedtuple
SceneRow = namedtuple('Row', ['scene', 'obs', 'mmsi', 'start', 'end', 'fps', 'tag'])
SceneRow.__new__.__defaults__ = (None, None, None, None, None, None, None)

## ship xysc
TrackRow_ship_gt = namedtuple('Row', ['frame', 'obs', 'mmsi', 'x', 'y', 's', 'c'])
TrackRow_ship_gt.__new__.__defaults__ = (None, None, None, None, None, None, None)

## ship xysc
TrackRow_ship_obs = namedtuple('Row', ['frame', 'obs', 'mmsi', 'x', 'y', 's', 'c'])
TrackRow_ship_obs.__new__.__defaults__ = (None, None, None, None, None, None, None)